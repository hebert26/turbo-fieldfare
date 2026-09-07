# Efficient testing goals and Gemma context

Read for goals, progress, retries, prompts, thought history, compaction, or
conversation reuse. This is design knowledge for future assigned implementation.
It does not claim that a complete goal loop already exists.

## Define and verify the outcome

A testing goal combines requested outcomes, permitted scope, and evidence that
each outcome was reached. Parse enough to proceed and clarify only material
ambiguity. A successful tool call, visited screen, generated plan, or model
statement alone does not establish task completion.

Keep the smallest useful structured task record in the host. Track each outcome
as pending, in progress, verified, failed, or blocked, with relevant observation
evidence and unfinished work. Record partial success honestly. A screen can be
revisited for valid reasons, such as adding another item. Judge progress through
state changes and remaining outcomes rather than screen identity alone.

The cycle is: present the goal and current facts, let Gemma choose one action,
execute through the existing bridge, observe its result, and update verified
progress before the next decision. Keep orchestration generic across apps.
User prompts can name app sections, ordering, and sample data. Do not encode
NestMind routes or silently choose the next application action for Gemma.

## Correct mistakes without unsafe retries

Distinguish malformed proposals never sent, pre-dispatch refusals, verified
actions, and unknown delivery. A permitted correction supplies the exact
failure and useful fresh choices. Gemma can reconsider accessibility facts
and request an image when ambiguity matters. Never automatically replay input
whose delivery is unknown. Follow VisionCapture's recovery contract.

Detect repeated action plus unchanged evidence plus absent goal progress.
Retain only bounded recent state needed for that check. Arbitrary call counts
do not define task completion. Preserve cancellation, resource limits, and
explicit stops for real blockers or repeated ineffective behavior. Report the
unfinished outcome instead of claiming success or spinning forever.

Long thinking can dominate task time despite acceptable decode speed. Preserve
the assigned thinking mode, expose actual generation progress, and distinguish
useful work from a reasoning/action cycle. A proposed thinking budget or
interruption policy needs explicit semantics for incomplete output, model turn
boundaries, and resumption. Never execute a partial tool call or fabricate an
answer when generation stops.

## Google prompt contract

Source checked 2026-09-07:
[Gemma 4: managing thought context](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4#managing-thought-context-between-turns).

Remove prior raw thoughts before the next ordinary conversation turn. Preserve
thoughts across function calls within the same model turn. Google recommends
concise reasoning summaries as ordinary text for longer tasks. Thinking is a
conversation setting. Thinking instructions and tool definitions belong in one
system turn. Preserve native tool delimiters and string encoding. The
tool-response opener is a generation stop boundary. Consult the linked examples
when changing the renderer, parser, or stopping behavior.

## Apply the contract without corrupting context

A tool call is not automatically a new ordinary turn. Trace actual rendered
boundaries before removing thoughts. One goal may involve many tool calls
within one model turn. Stripping after every tool result violates Google's
exception. A display-only change cannot reduce retained inference state.

Keep the user goal, factual progress, execution results, and model reasoning
distinct. A reasoning summary can preserve an unresolved decision or avoid
repeating a failed approach. It cannot prove that an action worked. Never
synthesize hidden reasoning or manufacture a completed task record.

Separate the visible transcript, canonical execution records, and compact
decision input. Give Gemma the goal, relevant verified progress, current screen
facts and selections, permitted actions, last result, and unresolved constraints.
Load protocol guidance only when needed. Avoid repeatedly injecting complete MCP
skills, duplicate trees, stale handles, and historical boilerplate. Preserve
facts needed to explain refusals, uncertain delivery, or user-data constraints.

Count the fully rendered input with the model tokenizer, including tools,
images, retained thoughts, and an output reserve. Use measured context and
memory budgets. A larger window does not make repeated content free.
Bound image retention and avoid repeating old full-resolution payloads.
Image eviction must respect live inference use.

Retained K/V represents the exact tokens and image embeddings processed.
Editing old text cannot make existing K/V correct for the new prompt. Reuse
only a verified matching prefix supported by the cache, or rebuild safely.
Sliding-window state can prevent rewinding even when an earlier textual prefix
matches. Plan necessary compaction at valid boundaries, preserve task evidence,
and measure re-prefill cost. Avoid rewriting the entire prefix every action
merely to update a small progress note.

## Production entry points

Inspect Sources/TurboFieldfareApp/Core/Tools/VisionCaptureToolLoop.swift,
VisionCaptureScreenFacts.swift, and VisionCaptureToolDefinitions.swift.
Follow Core/Inference/AppConversation.swift and Core/State/AppModel.swift for
host history and lifecycle. Runtime entry points under Sources/TurboFieldfare/
are Runtime/Generation/MultimodalConversation.swift, RawCompletion.swift,
and Runtime/Vision/MultimodalPromptRenderer.swift. Check actual template output,
stopping, token lineage, and completed-turn semantics, not just displayed chat.
