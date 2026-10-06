# Qwen team routing

Folder index: [[turboCharge/Index]].

Updated 2026-09-23. Use this staffing record with the [tracker](tracker-qwen3.6-35b-a3b-official-2026-09-15.md) and [implementation document](implementation-qwen3.6-35b-a3b-official-2026-09-15.md). It replaces the older Pi, Terra, Flash, DeepSeek and Grok recommendations, which remain in [version-1 history](history/version-1-before-bf16-20260923T180049Z/team-routing.md).

| Responsibility | Model | Reasoning | Boundary |
|---|---|---|---|
| Coordination and final acceptance | Astra / Main | Current task setting | Own scope, dependencies, assignments, evidence review and the human-facing page. |
| Production implementation and design | GPT-6 Sol | high | Own a bounded set of named production files. For the current request, own the tracker and implementation document only. |
| Independent tests | GPT-6 Luna | xhigh | Own separately assigned test files and meaningful checks. Tests run only under the serial execution lease. |
| Source, evidence and documentation audit | GPT-6 Luna | max | Read independently, report incorrect completion claims and verify revised documentation. No production edits. |

The user authorized up to 10 agents earlier. The current document task used Main, one Sol and two Luna reviewers: one for phase/source evidence, one for a bounded memory and historical-count check. Do not add agents merely to fill available slots.

Each assignment names its purpose, exclusive write paths, dependency, expected evidence and handback. Agents share the checkout. They preserve unrelated edits and never revert another owner's work. Main reviews results before accepting or reporting completion. A production writer's own test claim is not independent review.

## Parallel boundaries

Source inspection, reference-contract analysis, memory arithmetic and document review can run independently. Production and test writing can overlap only when their interface is agreed and file ownership is disjoint. Shared manifest, runtime, Metal binding and cache files have one writer at a time. Reuse existing agents for related work.

Builds, package tests, numerical references, model loading and inference are serial. A reference process must fully exit before the candidate starts. Keep the loopback server on `127.0.0.1`. No second model process or parallel benchmark. Each future model command must satisfy repository preflights.

## Current scope

Documentation alignment only. Sol revises the original-BF16 plan, Luna audits the code and evidence, Main integrates the HTML and handoff. Implementation and the goal remain paused. Pi, spoken responses and VoiceOver stay off. Historical version-1 completion is not original-BF16 acceptance.
