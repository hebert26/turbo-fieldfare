# VisionCapture evidence at stop

Inventory time: 8 September 2026. Review window begins approximately **2026-09-07 23:21 UTC**.

This is a read-only handoff. The VisionOS repository is on `codex/live-test-action-replay` at `96dab6ca6cbd0a792d78690177330f058ee172a7`. That commit predates the review window, and `git log --since=2026-09-07T23:21:00Z` contains no commit. The working tree is dirty. Git therefore proves current file contents, but does not prove who made each uncommitted change or when it was made.

## Earlier Phases 1–4 baseline

The saved evidence places these changes before the review window:

- Release responses omit private diagnostic prose while Debug retains it. All `execute` result exits use one final projection, and Release removes duplicate proof text without changing the structured verdict, delivery, recovery, session, choices or image blocks. Exact VisionCapture files: `VisionCapture/Package.swift`, `VisionCapture/Sources/HTTPServer/MCPRequestHandler.swift`, `VisionCapture/Sources/HTTPServer/MCPRequestHandlerHarness.swift`, `VisionCapture/Sources/HTTPServer/Tools/HTTPExecute+ComputerUseEvidenceDiagnostics.swift`, `HTTPExecute+ComputerUseLane.swift`, `HTTPExecute+PointerClick.swift`, `HTTPExecute+ResponseFormatting.swift`, `HTTPExecute+ResponseFormattingSupport.swift`, and the untracked `HTTPExecute+OutgoingResponseProjection.swift`.
- Release packaging verifies compiler conditions, required response sources, optimized Release modules and executable identity before and after staging. Exact files: `VisionCapture/scripts/build-app-only.sh`, `VisionCapture/scripts/deploy-app.sh`, and the untracked `VisionCapture/scripts/verify-concise-release.py`.
- The actual packaged comparison ran on 7 September at 21:19–21:20 UTC. Release and Debug returned byte-identical 39,074-byte public schemas and the same validation error. Release returned 284 bytes without the private harness. Debug returned 546 bytes with it. The installed Release executable was SHA-256 `fa4d7822eea5137346299b8fa3049b01d218433e018eb3d4f109d330d925f8c6` and was restored after the comparison.
- Phase 3 choice binding, changed-only hints, screenshot/read pairing and swipe handling live primarily in TurboCharge. The installed image-pair correction completed at 21:54 UTC on 7 September, also before this window. It delivered one real image, current accessibility facts and 13 fresh choices, then produced a completed answer. That receipt does not establish every VisionCapture action or image path.

Evidence: [Phases 1–4 archive](../../Project-files/active/mcp-concise-responses/evidence/20260907-phases1-4/README.md), [acceptance map](../../Project-files/active/mcp-concise-responses/evidence/20260907-phases1-4/acceptance-map.md), [packaged HTTP receipt](../../Project-files/active/mcp-concise-responses/evidence/live-20260907T2015/packaged-http/receipt.md), [installed identities](../../Project-files/active/mcp-concise-responses/evidence/live-20260907T2015/installed-builds.md), and [image-pair correction](../../Project-files/active/mcp-concise-responses/evidence/image-pair-correction-20260907/README.md).

## Current VisionOS changes whose window timing is unproved

These hunks are present now but are outside the archived Phase 1 production patch. No matching build, installed-binary identity or live acceptance receipt was found in the reviewed handoff evidence. Treat them as **source only and unverified** until their provenance is recovered:

- `VisionCapture/Sources/Interaction/ScreenDescriber+Payload.swift`, `VisionCapture/Sources/Interaction/ScreenDescriber.swift`, and `VisionCapture/Sources/Core/Engine/DescribeScreenTransportFormatter.swift` add explicit field-value publication. Secure fields publish `value_status: secure` without values or hashes. Editable fields distinguish available empty values, omitted values, unavailable values, placeholder ambiguity and redaction. Switch values remain available when ordinary values are omitted. Transport summaries avoid using secure values as labels.
- `VisionCapture/Sources/SimulatorFrameHelper/SimulatorFrameFrameworkProvenance.swift` adds macOS `26.6.2` build `25G83` to the closed supported-host list while retaining the existing Xcode and framework provenance checks.

The full VisionCapture working tree also still contains the earlier concise-response and packaging files listed above. Their dirty or untracked status does not make them new work inside this window because the pre-window receipts already record those behaviors and builds.

## Behavior observed during the review window

Captures 15–18 exercised the installed public MCP repeatedly on the same selected iPhone 17 simulator. Saved results show exact-device `APP_NOT_INSTALLED` refusal without mutation, successful install, `foreground_ready` launch/attachment, current accessibility reads, one actual screenshot/read pair, safe rejection of an expired target, native permission-alert choices and continued reads after an inconclusive transition. These runs verify those observed calls only. They do not reconnect the current dirty VisionOS source to the installed binary or verify the pending field-value and macOS-host additions.

Capture 18 ended after the owner stopped the exploration. The next conversation contained only `continue`; the app stated that earlier turns were outside model context, and the final answer asked what task to perform. No automatic checkpoint committed. This is the highest-priority continuity failure. It is in the TurboCharge conversation boundary and handoff path rather than a proved VisionCapture server defect.

Evidence: [capture 15](../experiments/phase5-runtime-measurement-20260907/capture-15/README.md), [capture 16](../experiments/phase5-runtime-measurement-20260907/capture-16/README.md), [capture 17](../experiments/phase5-runtime-measurement-20260907/capture-17/README.md), and [capture 18](../experiments/phase5-runtime-measurement-20260907/capture-18/README.md).

## Open MCP and integration evidence

- One transient computer-use capture error occurred in capture 18 and cleared after rereading the same app. The saved README does not classify the error, so its cause remains unknown.
- Packaged Debug/Release HTTP testing covers the public schema and one validation-error path. It does not run every success, alert, preview, recording, permission, image or recovery response family through both packaged binaries.
- Capture 17's expired target was blocked and current choices were returned. Capture 16's permission-alert and contradicted-transition cases also continued through guarded recovery. These are successful safety observations, not unresolved replay defects.
- The repeated-thought recovery event-identity defect was fixed in TurboCharge and the fixed app/helper were built, installed and started. A live repeated-thought recovery has not yet exercised that installed fix. Capture 18 did not trigger it.
- Stop/resume loses the original task and useful history when continuation starts a new conversation. Capture 18 proves the resulting loss of context, while the exact UI sequence and root cause still require investigation.

The TurboCharge receipt is [performance-compaction/implementation-receipt.md](../experiments/phase5-runtime-measurement-20260907/performance-compaction/implementation-receipt.md).

## Pending source-only continuity change

`Sources/TurboFieldfareApp/Core/Tools/AgentTaskCheckpoint.swift` currently has SHA-256 `86ee29340a9862b014df40282e5c9aa6dc119126c16c93a0e493930c0aeebc47`. It retains up to three recent unique sanitized historical observations, marked `historical_only`, within a 6,144-byte cap. The current decision packet remains the only actionable source. Parser and diff checks were recorded, but this change was **not built, installed or live verified** before the stop.

The next investigation should start with Stop/resume preserving the original task and checkpoint state. Current VisionOS field publication and host-provenance changes must remain classified as source only until a matching build and live receipt identify them.
