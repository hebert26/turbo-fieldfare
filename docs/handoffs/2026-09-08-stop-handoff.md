# TurboCharge handoff

Work stopped. Navigation improved, but reliable interruption, context continuity and full exploration remain unfinished.

## Read first

At 19:21 UTC (20:21 London), Hebert asked for this handoff and an end to work. Gemma was already idle. The app and loaded model were left intact. The owned sampler and reliability schedule were stopped.

**Next priority: let the user stop exploration, give a new instruction and retain the task context.** This document records remaining work. It does not start it.

Coverage: approximately 23:21 UTC on 7 September to 19:21 UTC on 8 September 2026. Earlier accepted work is identified separately. Saved receipts are more precise than broad phase labels. Dirty files alone do not prove authorship or when a change happened.

## 1. What was achieved

Gemma repeatedly completed onboarding and profile work and reached todos and bookmarks. Several runs showed the requested todo names and bookmark entries. These are partial journeys, not proof that every requested tag, search, video and playlist task completed.

TurboCharge now has host-managed task checkpoints, retained image/history handling, clearer progress and request/response visibility, and several narrow recovery corrections. Automatic compaction was observed once reducing the active context and resuming a read.

Important failures remain: long repeated thinking, recovery that has not yet succeeded in a live accepted case, an app text-layout freeze, and loss of task continuity after stopping/reloading. The latest run ended with Gemma asking what task to perform.

No Gemma weights were changed. No sustained speed gain or 25 tokens/second result was established. The tested Metal candidate is not installed. The owner's latest priority is reliability and compaction; speed experiments are parked.

## 2. Highest priority: Stop and resume lose the task

**Owner report:** once exploration starts, stopping does not provide a useful way to give Gemma a new instruction. The owner has to unload and reload, which loses the previous conversation context.

**Observed:** the chat displayed “Earlier turns are no longer in the model’s context.” The new input was only “continue”. Gemma answered: “I'm ready to proceed. Please let me know what task you would like me to perform or provide more instructions so I can continue.”

Capture 18 records original conversation `B84D068D-B3CF-4DF5-9F78-6D369E40B9BE` from 19:03:48–19:18:11 UTC, then `C35D041F-7DFC-4757-9592-D34F792F5C65` from 19:18:53–19:21:05. Final context was 2,263 tokens. There was no committed automatic checkpoint in this run.

This confirms missing task context in the new conversation. The trace and UI alone do not establish the exact user action sequence or root cause. Earlier successful Stop checks did not cover this entire flow.

**Required future outcome:** Stop settles or cancels current work safely, preserves the host task record and relevant history, and enables a new instruction without unloading. If a model reload requires rebuilding context, the original task and verified progress must be restored explicitly. An uncertain app action must never be replayed.

Start the source review at `AppModel.swift`, `VisionCaptureToolLoop.swift`, `AgentTaskCheckpoint.swift`, `DecodeServiceInferenceClient.swift` and `MultimodalConversation.swift`. Compare Stop, generation cancellation, unload and new-conversation paths before proposing a change.

[Capture 18 closeout](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-18/README.md) · [Exact model input/output trace](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-18/model-trace.jsonl)

## 3. TurboCharge changes during the work

| Area and files | Change and evidence limit |
| --- | --- |
| Tool loop — `Core/Tools/VisionCaptureToolLoop.swift`, `VisionCaptureScreenFacts.swift`, `VisionCaptureToolDefinitions.swift`, `VisionCaptureMCPClient.swift` | Shorter model facts, recovery feedback for proposals that were not sent, current/historical state handling, no-progress warnings and restricted visual refresh. Several later failures show recovery coverage remains incomplete. Preserve MCP refusals and uncertain-delivery restrictions. |
| Host memory — `AgentTaskCheckpoint.swift`, `RealInferenceClient.swift`, `MultimodalConversation.swift` | Task/action history, historical lookup and image retention across replacement; capacity and performance checkpoint assessment. Capture 17 proves automatic replacement and resumed read, not complete task continuity. |
| Repeated thinking — `StructuredAssistantDecoder.swift`, `AppInferenceError.swift`, `DecodeServiceInferenceClient.swift`, `DecodeProtocol.swift`, service `Entry.swift` / `DecodeServiceOutbox.swift` | Detect sustained exact repetition, preserve a settled pending tool result, restrict recovery and avoid replay. Capture 17 exposed missing conversation identity in the service terminal. Identity fix installed at 18:52 UTC. Successful live recovery is still pending. |
| Progress and chat — `AppModel.swift`, `AgentInferenceTrace.swift`, `VisionCaptureActivity.swift`, `LiveGenerationPreviewText.swift`, `OutputPaneView.swift`, `InstructionTranscriptDocumentController.swift` | Bounded thinking display, following/copy controls, trace/status visibility and compaction progress. A later AppKit layout stall required a narrow geometry correction. Capture 17 remained responsive beyond the earlier failure, but broad freeze-free reliability is unproved. |

App-relative files above are under `Sources/TurboFieldfareApp/`. Runtime, protocol and service files are under their corresponding `Sources/` modules.

Generic repeated instructions were moved out of per-result facts where standing developer instructions already covered them. Four sites were shortened, preserving current facts and safety restrictions. Saved captures showed 8,081 and 9,236 payload bytes removed in bounded comparisons. These are byte reductions, not measured speed gains.

[Instruction deduplication receipt](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/prompt-instruction-deduplication.md) · [Repeated-thinking recovery](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-15/repeated-thinking-recovery.md) · [Text-layout correction](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-16/chat-layout-geometry-correction.md)

## 4. VisionCapture changes and outstanding MCP issues

The component inventory is maintained in the linked evidence note. It distinguishes earlier Phases 1–4 work, source changes in this time window, installed evidence and unresolved issues. Do not attribute every dirty VisionCapture file to this session.

**Earlier baseline:** concise Release responses, Debug diagnostics, one final outgoing projection and packaging checks were already exercised before this 20-hour window. On 7 September at 21:19–21:20 UTC, packaged Release/Debug returned identical public schemas; the validation response was 284 bytes in Release and 546 bytes in Debug. This is narrow coverage, not all response families.

**Current source, timing and installation unproved:** `ScreenDescriber+Payload.swift`, `ScreenDescriber.swift` and `DescribeScreenTransportFormatter.swift` add explicit empty/omitted/secure/redacted field-value states. `SimulatorFrameFrameworkProvenance.swift` adds macOS 26.6.2 build 25G83 to the supported-host list. No matching build/live receipt was found for these hunks. Preserve them as source-only until verified.

VisionOS is on `codex/live-test-action-replay`, commit `96dab6ca6cbd0a792d78690177330f058ee172a7`. There are no commits inside the review window. The dirty tree cannot establish when or by whom each edit was made.

[VisionCapture file-by-file handoff](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/handoffs/2026-09-08-stop-handoff-visioncapture-evidence.md)

VisionCapture remains the source of truth for action results, current capabilities, refusals and uncertain delivery. TurboCharge translates those facts for Gemma. It must not invent success or bypass a terminal restriction.

Retain the review items around misleading/incomplete screen summaries, unnamed or duplicate controls, stale cached targets and multiple devices in parallel. These need evidence-led follow-up, not application-specific navigation code.

The missing activity notch was resolved by the owner identifying their own toggle. No notch code change was made for that report. The simulator-pixel checkbox controls preview permission, not notch visibility.

## 5. What changed for Gemma

Gemma receives an app-independent navigation interface, compact current facts and choices, supported screenshots, a standing instruction set and host-maintained task history. Internal device/session/grant handling belongs to TurboCharge and VisionCapture. The model chooses the navigation action.

The same task prompt asked for onboarding, profile, tagged todos, tagged bookmarks, real videos/playlists and search, while avoiding the Assistant. Root prepared fresh installations and observed results. Individual live taps were chosen by Gemma unless a receipt explicitly records a diagnostic intervention.

Images were actually supplied in recorded runs, including capture 18 input 2. Image support is therefore more than an advertised tool. Gemma still did not reliably request images when confused; one initial screenshot does not prove sustained visual recovery.

Thinking was enabled at the owner's request. It produced useful decisions but also long exact repetition. No fine-tuning, model weight change or fixed set of “navigation experts” was introduced. We cannot infer details of Gemma's training from these tests.

## 6. Compaction: implemented, observed, still limited

The host builds a replacement context from the original goal, recorded actions, retained history and actual images. It resets and rebuilds the same model conversation at a settled tool boundary. Gemma is not asked to write a summary.

The installed optional performance trigger requires three completed non-history tool decisions, at least 128 generated tokens, latest context of at least 20,480 tokens, and weighted generation below 15 tokens/second. It waits six completed decisions before another performance trigger. Capacity protection takes priority.

A replacement must fit the configured reserve and save at least 4,096 tokens or 20%, whichever is larger. If it does not, the existing state remains. The UI reports assessment, rebuild and outcome.

**Capture 17:** settled input fell from 20,989 to 8,001 tokens, saving 12,988. The checkpoint retained the original task, 52 execution records, 28 history references and one actual image. Rebuild took 162 seconds. A read resumed successfully.

**Limit:** the first resumed decision was 9.77 tokens/second versus 12.35 in the pre-trigger window. Different decisions make this an uncontrolled comparison. It proves neither speed gain nor reduced overall task time. Long thinking can still consume context before a safe boundary.

A new, unbuilt source change adds up to three recent historical observations, within 6,144 UTF-8 bytes, to make prior facts easier to use. Capture 17's example adds 4,542 bytes. It excludes executable IDs and marks these facts historical. It may help continuity, but its benefit is unverified.

[Exact policy, build receipt and pending continuity change](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/performance-compaction/implementation-receipt.md)

## 7. Experiment ledger

Each link contains the detailed commands, settings, timings, failures and protocol deviations. Some early captures establish baseline context for this roughly 20-hour window. Later conclusions supersede optimistic interim notes.

| Capture | Result and limitation |
| --- | --- |
| [1. Baseline and counters](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-1/README.md) | Long context baseline and measurement collection. Useful numeric evidence, no sustained 25 tokens/second result. |
| [2. Tool image cancellation](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-2/README.md) | Cancelled image-related answers and an incomplete closing footer. The failed request remains failed. |
| [3. Image memory](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-3/README.md) | Completed image request. Expert scratch release about 3.003 GiB, 960 slots refilled, image prefill 40.533 s. Logical read bytes are not physical disk traffic. |
| [4. Cancellation receipts](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-4/README.md) | Installed cancellation delivery and measurement-flush corrections. Acceptance is limited to those recorded checks. |
| [5. Chat responsiveness](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-5/README.md) | Bounded 8 KiB thinking preview, manual scroll pauses following, Follow Latest and copy support. Stop measured 1.530 s in that check, not proof for every exploration state. |
| [6. Image build recovery](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-6/README.md) | Installed image-path/build corrections and a completed single-image case. This does not establish all screenshot or cancellation paths. |
| [7. Checkpoint replacement](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-7/README.md) | Forced conversation replacement and resumed verified field entry. Full journey still incomplete. |
| [8. Checkpoint deduplication](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-8/README.md) | Reduced repetition in checkpoint material. Follow the linked receipt for exact retained content. |
| [9. Task history](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-9/README.md) | Host task record and bounded local historical observation lookup. History access exists, but the model may fail to request it. |
| [10. Unknown tool name](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-10/README.md) | 76 attempts, 13.2897 tokens/s, context 31,850. Onboarding/profile, three tagged todos and two bookmarks observed. Stopped on unknown tool name. Video/search and complete bookmark tagging unverified. |
| [11. Historical image confusion](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-11/README.md) | Detected confusion between historical image and current screen. Cancelled; ordering correction followed. |
| [12. Image ordering and loop](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-12/README.md) | Forced checkpoint 7,821 → 6,640 tokens. Original task, 24 action records, 14 history references and one image retained. Repeated Todos taps revealed no-progress accounting weakness. |
| [13. Recoverable malformed call](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-13/README.md) | Unknown-name proposal corrected without app input. 86 attempts, 12.2705 tokens/s, context 43,495. Later unchanged observations exposed a dropped warning and visual-refresh gap. |
| [14. Expired targets](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-14/README.md) | Ended after three expired target proposals were locally rejected. 71 attempts, 12.81075 tokens/s, context 32,781. No automatic checkpoint. |
| [15. Repeated thinking](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-15/README.md) | Long repetition caused a stop. Three todo names and Shopping List evidence existed, but the full goal did not complete. Repetition detector/recovery implemented afterward. |
| [16. App freeze](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-16/README.md) | Main thread sampled in text layout while decode helper was idle. App memory rose 398 → 435 MiB. Exact owned app terminated after normal Quit failed. Narrow geometry correction followed. |
| [17. Automatic compaction](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-17/README.md) | 20,989 → 8,001 tokens, 162 s rebuild, then a resumed read. Later repetition recovery failed because the service omitted conversation identity. Fixed and installed, successful live recovery still unproved. |
| [18. Final run and context loss](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-18/README.md) | Fresh test began 19:03 UTC. Later a new conversation contained only “continue”; Gemma asked what task to perform. Owner requested stop at 19:21 UTC. No full journey or automatic checkpoint in this run. |

Separate Metal investigation tested a query-register candidate: 26 direct numerical cases and 24 production-wrapper cases matched their baselines. Isolated timing across nine shapes and 432 samples showed about 16% median attention GPU reduction. That is not whole-model speed. Baseline kernels were restored for the live app.

Full-model short/long comparison fixtures were prepared but not run to acceptance. No Metal candidate was adopted. Server numeric-footer changes remain separate unbuilt work.

[Metal candidate](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/query-register-candidate.md) · [Production selection evidence](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/query-register-production-selection.md) · [Timing scope](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/query-register-timing-preparation.md) · [Unrun full-model fixtures](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/server-comparison-fixtures/README.md) · [Memory report](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/memory-measurements.md) · [Prefill report](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/prefill-measurements.md)

## 8. Installed versus source-only

**Installed:** release app and decode helper built, signed, installed and launched using `./script/build_and_run.sh --verify`, exit 0, at approximately 18:52 UTC. App build 33.40 s; helper build 4.47 s. Source base commit `9a33992fe246e2c282f2c6b3c14238e9a17f09f5` plus recorded dirty changes.

```text
App SHA-256
6c8bc57b36cdf9f5ae55f1d6e10b1d85fefdfbf9b17ce90e3e029f70d8eb3e13
Decode helper SHA-256
6d18dfe0f3fb3737a27cefd6605f74422cd0398017567c0c562c8708ad73ccd9
```

Includes automatic compaction, prior chat corrections, repeated-thinking recovery machinery, conversation-identity fix and instruction deduplication. Installed at `/Applications/TurboFieldfare.app`.

**Source only:** the latest `AgentTaskCheckpoint.swift` recent-observation addition, SHA-256 `86ee29340a9862b014df40282e5c9aa6dc119126c16c93a0e493930c0aeebc47`. Parsing and whitespace checks passed. Root review is incomplete. It has not been built, installed or tested live.

**Separate pending source:** `Sources/TurboFieldfareServer/Core/ServerInference.swift` and `ServerLog.swift` comparison/footer preparation. No accepted full-model comparison. Preserve these and other user changes.

## 9. Remaining work, in priority order

1. **Stop and new instruction:** reproduce the capture 18 continuity problem through the Stop/unload/new-conversation paths. Required result: new instructions retain the original goal and verified progress without unnecessary reload.

2. **Recovery acceptance:** exercise the installed identity fix during repeated thinking. Prove a useful resumed decision and no duplicate app action. Initial-user and checkpoint-generation repetition currently lack the tool-result retry path.

3. **Checkpoint continuity:** review the source-only recent-observation addition and verify retained goals, completed actions, images and safety state through a real automatic compaction.

4. **Finish the task:** complete a fresh visible NestMind journey covering the requested tags, todos, bookmarks, videos, playlists and search. Verify saved results. Avoid the Assistant.

5. **App reliability:** watch separate app and helper memory during a long transcript and Stop/resume. The prior freeze was app text layout, not active model decoding.

6. **MCP defects:** follow the component inventory and fix confirmed generic defects without adding NestMind-specific code.

These are handoff priorities, not work in progress. Performance and Metal comparisons remain parked under the latest owner decision. The earlier 25 tokens/second goal is unmet.

## 10. Environment, stop state and evidence

Machine: M2 Pro, 32 GiB RAM; macOS 26.6.2; Swift 6.3.2. Visible iPhone 17 simulator, iOS 26.5. No headless simulator was required. Target bundle `com.hebertgo.nestmind.debug`, simulator UDID `7BE1EC4B-8A9F-4C00-8A2C-D4321F9AB382`.

Settings: 64K context, 32 LFU slots, Thinking ON, temperature 0.2, Top-K 64, Top-P 0.95, prefill 128 ON, RDADVISE OFF. Text and vision packs remain in the owner's Application Support location. No model duplicate or download was needed for the latest runs.

At stop: TurboCharge PID 32400 and helper 32944 were idle/loaded. Final HUD approximately 11.3 tokens/second, 2.3K/66K context, 4.7 GB helper memory. Sampler PID 34623 was interrupted and confirmed absent; session 26586 exited 0. The reliability heartbeat was deleted. Previous renewed-work heartbeat had already been deleted. The existing incomplete goal was not marked complete.

At capture 18's early slowdown, rates of 7.38–12.31 tokens/second occurred around 5–6.4K context. App memory was about 232 MiB, helper 4,810 MiB; system memory was heavily used/compressed. Long history alone cannot explain that early slowdown. Neither a source regression nor memory causation was proved.

All fresh runs reset only the disposable debug app under the owner's instruction. Existing code changes and evidence remain intact. The engineer was stopped after the documentation inventory. No implementation, deployment or model run follows this handoff.

[Existing phase plan](/Users/dev-machine/dev/turbo-fieldfare-personal/Project-files/human/mcp-concise-responses-and-turbo-context-implementation-notes.html) · [Runtime evidence index](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/README.md) · [Last memory samples](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-18/model-top.txt) · [Exact last exploration task](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/phase5-runtime-measurement-20260907/capture-18/original-prompt.txt)

**Resume only when Hebert asks. Begin with Stop/resume context continuity.**

