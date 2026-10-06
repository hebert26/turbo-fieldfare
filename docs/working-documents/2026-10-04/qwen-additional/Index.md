# TurboCharge — document index

Entry point for this folder. Open the note that matches the question. Last organised: 2026-09-25.

Repository: `/Users/dev-machine/dev/turbo-fieldfare-personal`. The app source stays there. This folder holds the vault notes, the 15 September 2026 document archive, and one preserved checkout inside that archive.

Up: [[Assistant/Atlas/Projects/Inventory#Confirmed mappings|Atlas inventory]] · [[Assistant/Atlas/Projects/TurboCharge/Index|Atlas project index]].

- Qwen task status → [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Now|Tracker — Now]]
- Qwen design and acceptance → [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15|Implementation]]
- Gemma navigation measurements → [[turboCharge/Project-files/active/report|Phase 5 report]]
- What the 15 September cleanup moved → [[turboCharge/archive/2026-09-15-document-cleanup/archive-manifest|Archive manifest]]

## Current Qwen plan

Original-BF16 route for the downloaded official checkpoint. Gemma stays the default. Version 1 acceptance belongs to the history copies below.

[[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15|Tracker]] — Task status for the original-BF16 route. Updated 24 September 2026. The human page is generated from this note and the implementation.

- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Now|Now]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase mapping|Phase mapping]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Rules|Rules]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phases|Phases]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 1 - The exact official BF16 source identity is recognized|Phase 1 - The exact official BF16 source identity is recognized]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 2 - A source-backed descriptor is classified beside packed v2 and Gemma v1|Phase 2 - A source-backed descriptor is classified beside packed v2 and Gemma v1]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 3 - An independent official CPU reference is recorded|Phase 3 - An independent official CPU reference is recorded]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 4 - Historical quantization stays outside the selected route|Phase 4 - Historical quantization stays outside the selected route]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 5 - The local source validates offline without loading weights|Phase 5 - The local source validates offline without loading weights]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 6 - Existing BF16 shards register without a copied weight pack|Phase 6 - Existing BF16 shards register without a copied weight pack]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 7 - Verified original-source ranges are read safely|Phase 7 - Verified original-source ranges are read safely]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 8 - BF16 text matrices have a checked Metal contract|Phase 8 - BF16 text matrices have a checked Metal contract]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 9 - Full attention reads BF16 source projections|Phase 9 - Full attention reads BF16 source projections]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 10 - Linear attention reads BF16 source projections|Phase 10 - Linear attention reads BF16 source projections]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 11 - Selected BF16 experts share one atomic cache state|Phase 11 - Selected BF16 experts share one atomic cache state]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 12 - The source-backed text runner emits a token|Phase 12 - The source-backed text runner emits a token]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 13 - Source-backed turns commit or roll back atomically|Phase 13 - Source-backed turns commit or roll back atomically]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 14 - Official chat text uses the verified source tokenizer|Phase 14 - Official chat text uses the verified source tokenizer]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 15 - Incomplete source-backed tool output dispatches nothing|Phase 15 - Incomplete source-backed tool output dispatches nothing]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 16 - Still images use source-backed vision groups|Phase 16 - Still images use source-backed vision groups]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 17 - Decode service binds results to BF16 source identity|Phase 17 - Decode service binds results to BF16 source identity]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 18 - CLI runs verified original-precision requests|Phase 18 - CLI runs verified original-precision requests]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 19 - Loopback server serves verified BF16 chat|Phase 19 - Loopback server serves verified BF16 chat]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 20 - Mac app remembers the existing BF16 source and preserves Gemma|Phase 20 - Mac app remembers the existing BF16 source and preserves Gemma]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 21 - The already-downloaded official source is verified and registered|Phase 21 - The already-downloaded official source is verified and registered]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 22 - Real original-BF16 text matches the independent reference|Phase 22 - Real original-BF16 text matches the independent reference]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 23 - Real still images preserve the verified workflow|Phase 23 - Real still images preserve the verified workflow]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 24 - Actual memory and speed are measured on this Mac|Phase 24 - Actual memory and speed are measured on this Mac]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 25 - Opt-in Qwen can return safely to Gemma|Phase 25 - Opt-in Qwen can return safely to Gemma]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15#Changelog|Changelog]]

[[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15|Implementation]] — Design, phase dependencies, and acceptance evidence for the same route.

- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Status and selected outcome|Status and selected outcome]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Source evidence and fixed architecture|Source evidence and fixed architecture]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Independent reference and numerical acceptance|Independent reference and numerical acceptance]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Paired expert and reader failure contract|Paired expert and reader failure contract]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Dependency and execution rules|Dependency and execution rules]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Historical-to-current phase mapping|Historical-to-current phase mapping]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 1 - The exact official BF16 source identity is recognized|Phase 1 - The exact official BF16 source identity is recognized]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 2 - A source-backed descriptor is classified beside packed v2 and Gemma v1|Phase 2 - A source-backed descriptor is classified beside packed v2 and Gemma v1]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 3 - An independent official CPU reference is recorded|Phase 3 - An independent official CPU reference is recorded]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 4 - Historical quantization stays outside the selected route|Phase 4 - Historical quantization stays outside the selected route]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 5 - The local source validates offline without loading weights|Phase 5 - The local source validates offline without loading weights]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 6 - Existing BF16 shards register without a copied weight pack|Phase 6 - Existing BF16 shards register without a copied weight pack]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 7 - Verified original-source ranges are read safely|Phase 7 - Verified original-source ranges are read safely]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 8 - BF16 text matrices have a checked Metal contract|Phase 8 - BF16 text matrices have a checked Metal contract]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 9 - Full attention reads BF16 source projections|Phase 9 - Full attention reads BF16 source projections]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 10 - Linear attention reads BF16 source projections|Phase 10 - Linear attention reads BF16 source projections]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 11 - Selected BF16 experts share one atomic cache state|Phase 11 - Selected BF16 experts share one atomic cache state]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 12 - The source-backed text runner emits a token|Phase 12 - The source-backed text runner emits a token]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 13 - Source-backed turns commit or roll back atomically|Phase 13 - Source-backed turns commit or roll back atomically]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 14 - Official chat text uses the verified source tokenizer|Phase 14 - Official chat text uses the verified source tokenizer]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 15 - Incomplete source-backed tool output dispatches nothing|Phase 15 - Incomplete source-backed tool output dispatches nothing]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 16 - Still images use source-backed vision groups|Phase 16 - Still images use source-backed vision groups]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 17 - Decode service binds results to BF16 source identity|Phase 17 - Decode service binds results to BF16 source identity]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 18 - CLI runs verified original-precision requests|Phase 18 - CLI runs verified original-precision requests]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 19 - Loopback server serves verified BF16 chat|Phase 19 - Loopback server serves verified BF16 chat]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 20 - Mac app remembers the existing BF16 source and preserves Gemma|Phase 20 - Mac app remembers the existing BF16 source and preserves Gemma]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 21 - The already-downloaded official source is verified and registered|Phase 21 - The already-downloaded official source is verified and registered]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 22 - Real original-BF16 text matches the independent reference|Phase 22 - Real original-BF16 text matches the independent reference]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 23 - Real still images preserve the verified workflow|Phase 23 - Real still images preserve the verified workflow]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 24 - Actual memory and speed are measured on this Mac|Phase 24 - Actual memory and speed are measured on this Mac]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 25 - Opt-in Qwen can return safely to Gemma|Phase 25 - Opt-in Qwen can return safely to Gemma]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15#Unmeasured limits|Unmeasured limits]]

[[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/handoff|Handoff]] — Checkpoint identity, metadata registration, staffing, and the paused next task. Updated 24 September 2026.

- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/handoff#Fixed scope|Fixed scope]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/handoff#Design that the next agent must preserve|Design that the next agent must preserve]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/handoff#Staff and execution|Staff and execution]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/handoff#Next work|Next work]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/handoff#Rebuilding the human page|Rebuilding the human page]]

[[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/team-routing|Team routing]] — Who writes production files, who reviews, and which work may run together. Updated 23 September 2026.

- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/team-routing#Parallel boundaries|Parallel boundaries]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/team-routing#Current scope|Current scope]]

[[turboCharge/Project-files/human/qwen3.6-35b-a3b-official-implementation-notes.html|Human page]] — Generated view of the tracker and implementation. Title: “Qwen3.6-35B-A3B official integration — Implementation plan.” Rebuild it with the repository builder named in the handoff.

## Earlier continuation note

The current handoff is the restart note for the original-BF16 route. This older note records the blocked phase 17 continuation from the quantized plan.

[[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/stopped-recovery/handoff|Continuation handoff]] — Phases 1–16 were accepted on that plan, phase 17 was blocked, and phases 18–25 had not started.

- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/stopped-recovery/handoff#Program state|Program state]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/stopped-recovery/handoff#P17 decision record|P17 decision record]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/stopped-recovery/handoff#Repair and evidence state|Repair and evidence state]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/stopped-recovery/handoff#Continuation sequence|Continuation sequence]]

## Version 1 history

Frozen copies from 23 September 2026, before the original-BF16 revision. The live notes point here for the earlier wording.

[[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15|Version 1 tracker]] — Quantized-pack plan and its phase list.

- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Key|Key]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Now|Now]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phases|Phases]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Done-when rules|Done-when rules]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 1 - The exact official checkpoint is recognized|Phase 1 - The exact official checkpoint is recognized]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 2 - A v2 Qwen manifest loads without changing v1|Phase 2 - A v2 Qwen manifest loads without changing v1]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 3 - Tiny Qwen execution fixtures are reproducible|Phase 3 - Tiny Qwen execution fixtures are reproducible]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 4 - Tiny BF16 ranges convert deterministically|Phase 4 - Tiny BF16 ranges convert deterministically]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 5 - A local official snapshot validates offline|Phase 5 - A local official snapshot validates offline]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 6 - A complete Qwen pack is sized before writing|Phase 6 - A complete Qwen pack is sized before writing]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 7 - A tiny transformed pack resumes byte-identically|Phase 7 - A tiny transformed pack resumes byte-identically]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 8 - Qwen Metal bindings have one checked contract|Phase 8 - Qwen Metal bindings have one checked contract]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 9 - Qwen full attention matches the tiny oracle|Phase 9 - Qwen full attention matches the tiny oracle]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 10 - Qwen linear attention matches the tiny oracle|Phase 10 - Qwen linear attention matches the tiny oracle]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 11 - Qwen MoE matches the tiny oracle|Phase 11 - Qwen MoE matches the tiny oracle]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 12 - A tiny Qwen runner emits the expected tokens|Phase 12 - A tiny Qwen runner emits the expected tokens]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 13 - Qwen turns commit or roll back atomically|Phase 13 - Qwen turns commit or roll back atomically]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 14 - Qwen prompts match the pinned chat template|Phase 14 - Qwen prompts match the pinned chat template]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 15 - Incomplete Qwen tool output dispatches nothing|Phase 15 - Incomplete Qwen tool output dispatches nothing]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 16 - Qwen still images produce matching token rows|Phase 16 - Qwen still images produce matching token rows]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 17 - Decode service binds responses to loaded identity|Phase 17 - Decode service binds responses to loaded identity]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 18 - CLI runs verified Qwen requests|Phase 18 - CLI runs verified Qwen requests]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 19 - Loopback server serves verified Qwen chat|Phase 19 - Loopback server serves verified Qwen chat]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 20 - The app selects Qwen without moving Gemma|Phase 20 - The app selects Qwen without moving Gemma]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 21 - Authorized conversion writes Qwen beside Gemma|Phase 21 - Authorized conversion writes Qwen beside Gemma]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 22 - Real Qwen text behavior matches the quantized reference|Phase 22 - Real Qwen text behavior matches the quantized reference]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 23 - Real Qwen screenshots preserve verified workflow truth|Phase 23 - Real Qwen screenshots preserve verified workflow truth]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 24 - A controlled Gemma-Qwen comparison is recorded|Phase 24 - A controlled Gemma-Qwen comparison is recorded]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Phase 25 - Opt-in Qwen can roll back to Gemma|Phase 25 - Opt-in Qwen can roll back to Gemma]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15#Changelog|Changelog]]

[[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15|Version 1 implementation]] — Approved version 1 scope, then the 23 September pause before the BF16 revision.

- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Current requirement — implementation paused, 2026-09-23|Current requirement — implementation paused, 2026-09-23]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Historical approved scope - version 1|Historical approved scope - version 1]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Problem|Problem]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Topological waves|Topological waves]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Shared-file ownership transfers|Shared-file ownership transfers]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Risks|Risks]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Owners|Owners]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 1 - The exact official checkpoint is recognized|Phase 1 - The exact official checkpoint is recognized]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 2 - A v2 Qwen manifest loads without changing v1|Phase 2 - A v2 Qwen manifest loads without changing v1]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 3 - Tiny Qwen execution fixtures are reproducible|Phase 3 - Tiny Qwen execution fixtures are reproducible]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 4 - Tiny BF16 ranges convert deterministically|Phase 4 - Tiny BF16 ranges convert deterministically]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 5 - A local official snapshot validates offline|Phase 5 - A local official snapshot validates offline]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 6 - A complete Qwen pack is sized before writing|Phase 6 - A complete Qwen pack is sized before writing]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 7 - A tiny transformed pack resumes byte-identically|Phase 7 - A tiny transformed pack resumes byte-identically]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 8 - Qwen Metal bindings have one checked contract|Phase 8 - Qwen Metal bindings have one checked contract]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 9 - Qwen full attention matches the tiny oracle|Phase 9 - Qwen full attention matches the tiny oracle]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 10 - Qwen linear attention matches the tiny oracle|Phase 10 - Qwen linear attention matches the tiny oracle]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 11 - Qwen MoE matches the tiny oracle|Phase 11 - Qwen MoE matches the tiny oracle]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 12 - A tiny Qwen runner emits the expected tokens|Phase 12 - A tiny Qwen runner emits the expected tokens]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 13 - Qwen turns commit or roll back atomically|Phase 13 - Qwen turns commit or roll back atomically]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 14 - Qwen prompts match the pinned chat template|Phase 14 - Qwen prompts match the pinned chat template]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 15 - Incomplete Qwen tool output dispatches nothing|Phase 15 - Incomplete Qwen tool output dispatches nothing]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 16 - Qwen still images produce matching token rows|Phase 16 - Qwen still images produce matching token rows]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 17 - Decode service binds responses to loaded identity|Phase 17 - Decode service binds responses to loaded identity]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 18 - CLI runs verified Qwen requests|Phase 18 - CLI runs verified Qwen requests]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 19 - Loopback server serves verified Qwen chat|Phase 19 - Loopback server serves verified Qwen chat]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 20 - The app selects Qwen without moving Gemma|Phase 20 - The app selects Qwen without moving Gemma]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 21 - Authorized conversion writes Qwen beside Gemma|Phase 21 - Authorized conversion writes Qwen beside Gemma]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 22 - Real Qwen text behavior matches the quantized reference|Phase 22 - Real Qwen text behavior matches the quantized reference]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 23 - Real Qwen screenshots preserve verified workflow truth|Phase 23 - Real Qwen screenshots preserve verified workflow truth]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 24 - A controlled Gemma-Qwen comparison is recorded|Phase 24 - A controlled Gemma-Qwen comparison is recorded]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Phase 25 - Opt-in Qwen can roll back to Gemma|Phase 25 - Opt-in Qwen can roll back to Gemma]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Decisions|Decisions]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Waivers|Waivers]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Discovered work|Discovered work]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15#Open questions|Open questions]]

[[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/handoff|Version 1 handoff]] — Restart instructions and the phase 15 and 16 acceptance addenda.

- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/handoff#Current Phase 16 acceptance addendum — 2026-09-18|Current Phase 16 acceptance addendum — 2026-09-18]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/handoff#Current Phase 15 acceptance addendum — 2026-09-17|Current Phase 15 acceptance addendum — 2026-09-17]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/handoff#Current status addendum — 2026-09-17|Current status addendum — 2026-09-17]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/handoff#Read this first|Read this first]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/handoff#Phase 14 evidence that must remain intact|Phase 14 evidence that must remain intact]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/handoff#Binding correction contract|Binding correction contract]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/handoff#Immutable boundaries and safety rules|Immutable boundaries and safety rules]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/handoff#Pending Pi replay repair|Pending Pi replay repair]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/handoff#Current staffing and changed gates|Current staffing and changed gates]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/handoff#Codex-budget configuration|Codex-budget configuration]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/handoff#Cleanup and pane state|Cleanup and pane state]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/handoff#Safe restart and resume order|Safe restart and resume order]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/handoff#Copy-paste resume prompt for the original Main|Copy-paste resume prompt for the original Main]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/handoff#P15–P25 roadmap|P15–P25 roadmap]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/handoff#Canonical documents and evidence index|Canonical documents and evidence index]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/handoff#Current documentation-task boundary|Current documentation-task boundary]]

[[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/team-routing|Version 1 team routing]] — Earlier staffing presets.

- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/team-routing#Historical presets|Historical presets]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/team-routing#Future staffing decision|Future staffing decision]]
- [[turboCharge/Project-files/active/qwen3.6-35b-a3b-official/history/version-1-before-bf16-20260923T180049Z/team-routing#Historical phase routing|Historical phase routing]]

[[turboCharge/Project-files/human/history/qwen3.6-35b-a3b-official-version-1-20260923T180049Z.html|Version 1 human page]] — Saved generated page from that snapshot. Its visible heading says implementation remains paused.

## Gemma investigation

[[turboCharge/Project-files/active/report|Phase 5 report]] — 7 September 2026 measurements of Gemma navigation, expert reuse, screenshot refill, and remaining delay. The note recommends keeping the current expert cache and resolving the conversation-history contract before changing the runtime.

- [[turboCharge/Project-files/active/report#Understand the model before choosing changes|Understand the model before choosing changes]]
- [[turboCharge/Project-files/active/report#Measurement baseline and provenance|Measurement baseline and provenance]]
- [[turboCharge/Project-files/active/report#1. Actual expert reuse: aggregate evidence, identities still missing|1. Actual expert reuse: aggregate evidence, identities still missing]]
- [[turboCharge/Project-files/active/report#2. Screenshot cache release and refill: observed memory transition|2. Screenshot cache release and refill: observed memory transition]]
- [[turboCharge/Project-files/active/report#3. Memory at 64K/32 slots/Thinking ON|3. Memory at 64K/32 slots/Thinking ON]]
- [[turboCharge/Project-files/active/report#4. All-hit scheduling: possible shortcut, benefit unmeasured|4. All-hit scheduling: possible shortcut, benefit unmeasured]]
- [[turboCharge/Project-files/active/report#5. Prefill batches: existing grouping already avoids duplicate loads|5. Prefill batches: existing grouping already avoids duplicate loads]]
- [[turboCharge/Project-files/active/report#6. Remaining delay: full attention and tool time matter|6. Remaining delay: full attention and tool time matter]]
- [[turboCharge/Project-files/active/report#Commands, deviations and remaining work|Commands, deviations and remaining work]]

The report links to `Project-files/active/google-deep-investigation-20260907/`. That directory is not in this folder.

## Archive — 15 September 2026 cleanup

[[turboCharge/archive/2026-09-15-document-cleanup/archive-manifest|Archive manifest]] — What moved into this batch, what stayed current, and which links were already broken before the move.

- [[turboCharge/archive/2026-09-15-document-cleanup/archive-manifest#Archived|Archived]]
- [[turboCharge/archive/2026-09-15-document-cleanup/archive-manifest#Retained current documents|Retained current documents]]
- [[turboCharge/archive/2026-09-15-document-cleanup/archive-manifest#Retained unresolved document|Retained unresolved document]]
- [[turboCharge/archive/2026-09-15-document-cleanup/archive-manifest#Baseline and link checks|Baseline and link checks]]
- [[turboCharge/archive/2026-09-15-document-cleanup/archive-manifest#Read-only model-comparison finding|Read-only model-comparison finding]]
- [[turboCharge/archive/2026-09-15-document-cleanup/archive-manifest#Missing at inventory time|Missing at inventory time]]
- [[turboCharge/archive/2026-09-15-document-cleanup/archive-manifest#Counts|Counts]]

### Superseded plans and checks

[[turboCharge/archive/2026-09-15-document-cleanup/repository/plan/report|Qwen integration analysis, 14 September 2026]] — Planning report for a second model family beside Gemma. The current tracker is the executable plan; this report is background.

- [[turboCharge/archive/2026-09-15-document-cleanup/repository/plan/report#1. Decision|1. Decision]]
- [[turboCharge/archive/2026-09-15-document-cleanup/repository/plan/report#2. Task contract|2. Task contract]]
- [[turboCharge/archive/2026-09-15-document-cleanup/repository/plan/report#3. Inspection baseline and conflict check|3. Inspection baseline and conflict check]]
- [[turboCharge/archive/2026-09-15-document-cleanup/repository/plan/report#4. Verified Qwen identity|4. Verified Qwen identity]]
- [[turboCharge/archive/2026-09-15-document-cleanup/repository/plan/report#5. Quantized source decision|5. Quantized source decision]]
- [[turboCharge/archive/2026-09-15-document-cleanup/repository/plan/report#6. Current TurboFieldfare findings|6. Current TurboFieldfare findings]]
- [[turboCharge/archive/2026-09-15-document-cleanup/repository/plan/report#7. Reuse, adapt, or add|7. Reuse, adapt, or add]]
- [[turboCharge/archive/2026-09-15-document-cleanup/repository/plan/report#8. Target design|8. Target design]]
- [[turboCharge/archive/2026-09-15-document-cleanup/repository/plan/report#9. Ordered implementation packages|9. Ordered implementation packages]]
- [[turboCharge/archive/2026-09-15-document-cleanup/repository/plan/report#10. Verification matrix|10. Verification matrix]]
- [[turboCharge/archive/2026-09-15-document-cleanup/repository/plan/report#11. Future comparison protocol|11. Future comparison protocol]]
- [[turboCharge/archive/2026-09-15-document-cleanup/repository/plan/report#12. Known unknowns|12. Known unknowns]]
- [[turboCharge/archive/2026-09-15-document-cleanup/repository/plan/report#13. Completion checklist|13. Completion checklist]]
- [[turboCharge/archive/2026-09-15-document-cleanup/repository/plan/report#Subsequent owner decision and source-preparation record (2026-09-15)|Subsequent owner decision and source-preparation record (2026-09-15)]]

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/qwen-ios-integration.html|Qwen for iOS testing — plan for review]] — Superseded 6 September 2026 MLX plan. The official-BF16 plan replaces it. [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/qwen-ios-integration.html|Compatibility symlink]] opens the same page.

`archive/2026-09-15-document-cleanup/Project-files/qwen-ios-integration` is a symlink to `active/qwen-ios-integration`. That target is not in the archive. The preserved checkout still has the 6 September tracker and implementation, linked under the checkout section.

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/turbocharge-evaluation-goal|Evaluation and performance goal]] — Owner direction from 4 September 2026: Gemma-driven iOS testing through VisionCapture, with a measured 25 token/s target. Updated 6 September 2026. The note records that target as unachieved.

- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/turbocharge-evaluation-goal#Objective|Objective]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/turbocharge-evaluation-goal#Sources and evaluation|Sources and evaluation]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/turbocharge-evaluation-goal#Architecture and performance constraints|Architecture and performance constraints]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/turbocharge-evaluation-goal#Execution state|Execution state]]

The goal links to `turbo-buffer-reuse-comparison-20260906.md` in the same folder. That sibling file is absent. The preserved checkout has it at [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/human/turbo-buffer-reuse-comparison-20260906|Buffer-reuse comparison]].

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/turbo-new-chat-memory-20260905|Completed chat memory check]] — 5 September 2026 check that New Chat cleared a finished Journey 5 transcript. The note records the footprint change and leaves the leak question open.

- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/turbo-new-chat-memory-20260905#Observed setup|Observed setup]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/turbo-new-chat-memory-20260905#Measurements|Measurements]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/turbo-new-chat-memory-20260905#Source finding and deployed correction|Source finding and deployed correction]]

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/gemma-visioncapture-mcp-implementation-notes.html|Gemma to VisionCapture MCP proof]] — Archived implementation notes for the Gemma navigation proof.

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/mcp-concise-responses-and-turbo-context-implementation-notes.html|Concise MCP responses and clearer Gemma context]] — Archived notes on shorter tool responses and the context sent to Gemma.

### Expert cache

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/expert-cache-implementation-notes|Expert cache implementation notes]] — 12 September 2026 summary. The review was complete; the sorting bypass and direct expert-ID lookup were still open in this note.

- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/expert-cache-implementation-notes#Current plan|Current plan]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/expert-cache-implementation-notes#What exists|What exists]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/expert-cache-implementation-notes#Remaining work: 0 of 4 tasks complete|Remaining work: 0 of 4 tasks complete]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/expert-cache-implementation-notes#Saved performance evidence|Saved performance evidence]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/expert-cache-implementation-notes#Source references|Source references]]

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/expert-cache-implementation-notes.html|Expert cache page, with the evidence]] — HTML beside the raw cache-work tree. Visible heading: “No unresolved choice.”

[[turboCharge/archive/2026-09-15-document-cleanup/docs/expert-cache-implementation-notes.html|Expert cache page, from docs]] — Archived copy from `docs/`. Visible heading: “Retain neither proposed runtime change.”

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/expert-cache-runtime-20260913/result|Expert-cache runtime result, 13 September 2026]] — Sorting bypass and reverse lookup were excluded. The queue shortcut stayed, and its speed benefit is recorded as unverified.

- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/expert-cache-runtime-20260913/result#Decision|Decision]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/expert-cache-runtime-20260913/result#Environment|Environment]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/expert-cache-runtime-20260913/result#Source states|Source states]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/expert-cache-runtime-20260913/result#128-slot results|128-slot results]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/expert-cache-runtime-20260913/result#First-response proxy and memory|First-response proxy and memory]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/expert-cache-runtime-20260913/result#Correctness and commands|Correctness and commands]]

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/expert-cache-runtime-20260913/sorting-bypass-invalid-stale-queue-binary/INVALID|Invalid sorting-bypass run]] — Marker that those outputs executed the queue-control binary and are excluded.

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/task-1.1-128-slot-d40af7e-vs-42d92e13-20260912T155039+0100/resumed-20260912T161141+0100/blocker|Task 1.1 blocker]] — 12 September 2026 stop. The approved `d40af7e` binary rejected `--expert-cache-slots 128` before loading a model.

Diagrams archived with the docs bundle:

- [[turboCharge/archive/2026-09-15-document-cleanup/docs/assets/expert-cache-current-vs-proposed.png|Expert cache, current versus proposed]]
- [[turboCharge/archive/2026-09-15-document-cleanup/docs/assets/expert-cache-lookup-planner.png|Expert-cache lookup planner]]
- [[turboCharge/archive/2026-09-15-document-cleanup/docs/assets/turbofieldfare-metal-architecture.png|Metal architecture]]
- [[turboCharge/archive/2026-09-15-document-cleanup/docs/assets/turbofieldfare-metal-architecture-v2.png|Metal architecture, later drawing]]

## Preserved checkout

`archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/` is a TurboFieldfare checkout kept with the cache evidence. `task-1.1-128-slot-d40af7e-vs-42d92e13-20260912T155039+0100/build-current/` and `build-d40/` are the comparison builds, including dependency checkouts. This index links the checkout's documents. Source, tests, and those build trees stay in the snapshot as files.

The checkout also contains its own `Project-files` copies of notes that are archived separately above. Two of those copies differ from the archived files:

- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/report|Checkout copy of the Phase 5 report]] — same measurements. Its diagnostic receipt path still points at `/Users/dev-machine/dev/VisionOS/`.
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/human/mcp-concise-responses-and-turbo-context-implementation-notes.html|Checkout copy of the concise-response page]] — the page before the archive link adjustment.

### Checkout documents

- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/README|README — TurboFieldfare]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/SYSTEM_DESIGN|System design]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/OPTIMIZATION_JOURNEY|Optimization journey]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/RUNTIME_CONTROLS|Runtime controls]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/BENCHMARKS|Benchmarks]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/COMMUNITY_BENCHMARKS|Community benchmarks]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/IMPLEMENTATION_REFERENCES|Implementation references]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/OPENAI_SERVER|OpenAI-compatible server]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/WORKFLOW_PRESETS|Workflow presets]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/experiments/EXPERIMENT_INVENTORY|Experiment inventory]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/experiments/summaries/01-model-install-and-expert-io|Summary 01 — model install and expert I/O]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/experiments/summaries/02-decode-moe-int4-and-router|Summary 02 — decode, MoE int4, and router]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/experiments/summaries/03-expert-cache-prediction-and-layout|Summary 03 — expert-cache prediction and layout]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/experiments/summaries/04-rdadvise|Summary 04 — RDADVISE]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/experiments/summaries/05-attention-and-kv-cache|Summary 05 — attention and KV cache]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/experiments/summaries/06-prefill|Summary 06 — prefill]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/experiments/summaries/07-fusions-head-and-orchestration|Summary 07 — fusions, head, and orchestration]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/experiments/summaries/08-sampling-tokenization-and-output|Summary 08 — sampling, tokenization, and output]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/experiments/summaries/09-validation-and-measurement-lessons|Summary 09 — validation and measurement]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/gemma-prompt|Gemma agent-mode prompt]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/openai-guide|OpenAI guide]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/pi-guide|Pi guide]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/AGENTS|Repository agent instructions]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/CONTRIBUTING|Contributing]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/SECURITY|Security]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/THIRD_PARTY_NOTICES|Third-party notices]]

### Stop handoff inside the checkout

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff|8 September 2026 stop handoff]] — Work stopped with navigation improved and interruption, context continuity, and full exploration still unfinished.

- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff#Read first|Read first]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff#1. What was achieved|1. What was achieved]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff#2. Highest priority: Stop and resume lose the task|2. Highest priority: Stop and resume lose the task]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff#3. TurboCharge changes during the work|3. TurboCharge changes during the work]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff#4. VisionCapture changes and outstanding MCP issues|4. VisionCapture changes and outstanding MCP issues]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff#5. What changed for Gemma|5. What changed for Gemma]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff#6. Compaction: implemented, observed, still limited|6. Compaction: implemented, observed, still limited]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff#7. Experiment ledger|7. Experiment ledger]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff#8. Installed versus source-only|8. Installed versus source-only]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff#9. Remaining work, in priority order|9. Remaining work, in priority order]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff#10. Environment, stop state and evidence|10. Environment, stop state and evidence]]

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff-visioncapture-evidence|VisionCapture evidence at stop]] — Read-only inventory of the VisionCapture tree at the 8 September stop.

- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff-visioncapture-evidence#Earlier Phases 1–4 baseline|Earlier Phases 1–4 baseline]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff-visioncapture-evidence#Current VisionOS changes whose window timing is unproved|Current VisionOS changes whose window timing is unproved]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff-visioncapture-evidence#Behavior observed during the review window|Behavior observed during the review window]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff-visioncapture-evidence#Open MCP and integration evidence|Open MCP and integration evidence]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff-visioncapture-evidence#Pending source-only continuity change|Pending source-only continuity change]]

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/docs/handoffs/2026-09-08-stop-handoff.html|Stop handoff page]] — HTML companion to the 8 September handoff.

### Phase 5 notes inside the checkout

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/cache-analysis/findings|Cache and prefill evidence]] — Read-only continuation. Recommends retaining the current expert selection, cache replacement, image residency, and prefill scheduling.

- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/cache-analysis/findings#1. Expert reuse: one new defensible bound|1. Expert reuse: one new defensible bound]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/cache-analysis/findings#2. Screenshot release and refill|2. Screenshot release and refill]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/cache-analysis/findings#4. All-cache-hit scheduling|4. All-cache-hit scheduling]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/cache-analysis/findings#5. Reuse inside prefill batches|5. Reuse inside prefill batches]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/cache-analysis/findings#6. Metal priorities and limits|6. Metal priorities and limits]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/cache-analysis/findings#Reproduction and limitations|Reproduction and limitations]]

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/journey-analysis/journey-findings|Saved journey measurements]] — Journey 27 and 28 evidence on full-attention cost and long tool requests.

- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/journey-analysis/journey-findings#Evidence and baseline|Evidence and baseline]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/journey-analysis/journey-findings#Generation and complete elapsed time|Generation and complete elapsed time]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/journey-analysis/journey-findings#Actual screenshot turn|Actual screenshot turn]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/journey-analysis/journey-findings#Recovery and tool timing|Recovery and tool timing]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/journey-analysis/journey-findings#Remaining GPU timing|Remaining GPU timing]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/journey-analysis/journey-findings#Commands, limitations and next measurement|Commands, limitations and next measurement]]

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/journey-analysis/new-user-thought-retention|New-user thought retention]] — 7 September 2026 source reading: a later user message in the same conversation keeps raw thought tokens.

- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/journey-analysis/new-user-thought-retention#Exact path|Exact path]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/journey-analysis/new-user-thought-retention#Display is a separate projection|Display is a separate projection]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/journey-analysis/new-user-thought-retention#Existing trace corroboration|Existing trace corroboration]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/journey-analysis/new-user-thought-retention#File hashes read during this check|File hashes read during this check]]

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/phase5-investigation/literature/google-model-foundation|Gemma 4 literature foundation]] — 7 September 2026 literature portion of Phase 5. Public sources are identified as mutable.

### 6 September Qwen-for-iOS plan inside the checkout

The archive symlink to this plan's folder has no target. These are the preserved copies.

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/tracker-qwen-ios-integration-2026-09-06|Qwen for iOS testing — tracker]] — 6 September 2026 tracker. Awaiting approval. The owner update says not to create a worktree yet.

- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/tracker-qwen-ios-integration-2026-09-06#Key|Key]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/tracker-qwen-ios-integration-2026-09-06#Now|Now]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/tracker-qwen-ios-integration-2026-09-06#Phases|Phases]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/tracker-qwen-ios-integration-2026-09-06#Done-when rules|Done-when rules]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/tracker-qwen-ios-integration-2026-09-06#Phase 1 - Qwen installs as a verified model pack|Phase 1 - Qwen installs as a verified model pack]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/tracker-qwen-ios-integration-2026-09-06#Phase 2 - Qwen reads the exact chat and tool history|Phase 2 - Qwen reads the exact chat and tool history]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/tracker-qwen-ios-integration-2026-09-06#Phase 3 - Qwen updates its recurrent memory on Metal|Phase 3 - Qwen updates its recurrent memory on Metal]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/tracker-qwen-ios-integration-2026-09-06#Phase 4 - Qwen attends to earlier tokens on Metal|Phase 4 - Qwen attends to earlier tokens on Metal]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/tracker-qwen-ios-integration-2026-09-06#Phase 5 - Qwen selects and streams the correct experts|Phase 5 - Qwen selects and streams the correct experts]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/tracker-qwen-ios-integration-2026-09-06#Phase 6 - Qwen generates text through the runtime|Phase 6 - Qwen generates text through the runtime]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/tracker-qwen-ios-integration-2026-09-06#Phase 7 - Qwen restores a conversation after an interrupted turn|Phase 7 - Qwen restores a conversation after an interrupted turn]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/tracker-qwen-ios-integration-2026-09-06#Phase 8 - Qwen understands screenshots in a conversation|Phase 8 - Qwen understands screenshots in a conversation]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/tracker-qwen-ios-integration-2026-09-06#Phase 9 - Existing clients load the selected model|Phase 9 - Existing clients load the selected model]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/tracker-qwen-ios-integration-2026-09-06#Phase 10 - The current iOS navigation loop accepts Qwen decisions|Phase 10 - The current iOS navigation loop accepts Qwen decisions]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/tracker-qwen-ios-integration-2026-09-06#Phase 11 - Qwen completes the recorded iOS test journeys|Phase 11 - Qwen completes the recorded iOS test journeys]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/tracker-qwen-ios-integration-2026-09-06#Changelog|Changelog]]

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06|Qwen for iOS testing — implementation]] — Implementation companion for that tracker.

- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Approved scope - version 1|Approved scope - version 1]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Problem|Problem]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Phase order|Phase order]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Risks|Risks]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Owners|Owners]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Phase 1 - Qwen installs as a verified model pack|Phase 1 - Qwen installs as a verified model pack]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Phase 2 - Qwen reads the exact chat and tool history|Phase 2 - Qwen reads the exact chat and tool history]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Phase 3 - Qwen updates its recurrent memory on Metal|Phase 3 - Qwen updates its recurrent memory on Metal]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Phase 4 - Qwen attends to earlier tokens on Metal|Phase 4 - Qwen attends to earlier tokens on Metal]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Phase 5 - Qwen selects and streams the correct experts|Phase 5 - Qwen selects and streams the correct experts]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Phase 6 - Qwen generates text through the runtime|Phase 6 - Qwen generates text through the runtime]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Phase 7 - Qwen restores a conversation after an interrupted turn|Phase 7 - Qwen restores a conversation after an interrupted turn]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Phase 8 - Qwen understands screenshots in a conversation|Phase 8 - Qwen understands screenshots in a conversation]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Phase 9 - Existing clients load the selected model|Phase 9 - Existing clients load the selected model]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Phase 10 - The current iOS navigation loop accepts Qwen decisions|Phase 10 - The current iOS navigation loop accepts Qwen decisions]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Phase 11 - Qwen completes the recorded iOS test journeys|Phase 11 - Qwen completes the recorded iOS test journeys]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Decisions|Decisions]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Waivers|Waivers]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Discovered work|Discovered work]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/active/qwen-ios-integration/implementation-qwen-ios-integration-2026-09-06#Open questions|Open questions]]

[[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/human/turbo-buffer-reuse-comparison-20260906|Buffer-reuse comparison, 6 September 2026]] — The candidate was slower in the short, medium, and long cases. Outputs matched, and the isolated source change was reverted.

- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/human/turbo-buffer-reuse-comparison-20260906#Measured cases|Measured cases]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/human/turbo-buffer-reuse-comparison-20260906#Environment and limits|Environment and limits]]
- [[turboCharge/archive/2026-09-15-document-cleanup/Project-files/human/cache-work/baseline-d40af7e/Project-files/human/turbo-buffer-reuse-comparison-20260906#Commands and full footers|Commands and full footers]]
