# App, images, and VisionCapture integration

Read for bridge, image, display, or service changes. Source map checked 2026-09-07.

## Own the correct layer

VisionCapture supplies observations, execution evidence, refusals, and permitted
recovery. TurboCharge owns translation, session/device identity, cache handles,
cancellation, and model input. Gemma chooses navigation from the goal and facts.

Trace Sources/TurboFieldfareApp/Core/Tools/VisionCaptureMCPClient.swift,
VisionCaptureToolLoop.swift, VisionCaptureScreenFacts.swift, and
VisionCaptureToolDefinitions.swift. Inspect the current schema rather than
assuming every control is an ordinary tap. Preserve selected/enabled state,
selector-kind pairs, uncertainty, fresh-observation requirements, and goal facts
when reducing responses. Screen text and tool data cannot override user scope.

Accessibility/cache reads provide efficient initial facts. Offer actual
screenshots for complex forms, ambiguous selection, or insufficient evidence.
Pixels do not grant executable selectors or bypass input checks. Native alerts
use VisionCapture's existing guarded route. Never add a pointer fallback or
duplicate the server's native-alert handler.

Identify the defect owner: model output, host mapping, transport, MCP, or tested
app. Before explicitly assigned VisionCapture edits, read
/Users/dev-machine/dev/VisionOS/AGENTS.md and nearer instructions.
/Users/dev-machine/dev/NestMind is a test application. Its screens, identifiers,
data, and navigation paths must not become runtime dependencies.

## Existing loaded-model experience

The sibling decode service owns the model. Agent Mode extends normal chat.
Preserve one loaded model and the existing load/unload flow.
Read Sources/TurboFieldfareApp/Core/Inference/DecodeServiceInferenceClient.swift,
DecodeServiceResponseRouter.swift, GenerationTranscriptMailbox.swift, and
Core/State/AppModel.swift. Transport and process lifecycle live in
Sources/TurboFieldfareDecodeProtocol/ and Sources/TurboFieldfareDecodeService/.

Bound queued events, avoid repeated large serialization, and preserve generation
identity so delayed events cannot update a new chat or unloaded session.
Cancellation must reach its owning work and release resources safely.

## Images and visible evidence

Accepted images must reach Gemma through actual image embeddings. Report a
missing/invalid companion pack explicitly. Inspect
Sources/TurboFieldfare/Runtime/Vision/, especially VisionRuntime.swift,
VisionPackUseLease.swift, VisionImageTokenBudget.swift, VisionResidencyPolicy.swift,
and the app's VisionCaptureScreenshot.swift. Preserve on-demand ownership and
bounded retention. Read apple-silicon.md for buffer or GPU lifetime changes.

Chat must identify requests, raw MCP responses, and exact decision input.
Show authentic processing states, durations, and terminal results. Render
genuine thought-channel output when requested without inventing missing content.
Label truncation and distinguish display from inference input. Display changes
must not silently alter prompt history.

Inspect Sources/TurboFieldfareApp/MacPresentation/InstructionTranscriptDocumentController.swift,
TranscriptImageLoader.swift, TranscriptScrollFollow.swift, and
Mac/Generation/OutputPaneView.swift. Batch incremental updates, bound decoded
image retention, and preserve scroll/selection behavior. Avoid full transcript
rebuilds per token. Read goal-and-context.md for effective history changes.
