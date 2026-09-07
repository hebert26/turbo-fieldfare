# Evidence for performance changes

Read for an assigned comparison or a performance claim. This does not authorize
a separate benchmark campaign. Coordinate with the live-run owner and follow
repository preflight rules.

## Hold the workload steady

The owner's comparison profile is 65,536 context tokens, 32 expert slots per
layer, LFU, temperature 0.2, Top-K 64, Top-P 0.95, prefill enabled with 128-token
chunks, and RDADVISE off. Confirm the assignment and actual request. These are
not upstream defaults. Record explicit changes, including thinking, vision,
prompt length, sampling seed, and output length.

Read docs/COMMUNITY_BENCHMARKS.md for community comparisons. Record deviations
for the owner's settings and app journeys. Freeze source, inputs, settings,
and model identity per candidate. Never edit inputs or build against another
worker's changing files during a comparison.

Trace the bottleneck and state expected benefit and memory/correctness cost
before editing. Use the smallest relevant comparison. Preserve complete useful
outputs, failures, and rejected candidates. Include representative long context
when claiming exploration benefits. Use warmups and enough repeated or
restored-baseline evidence to distinguish machine variation. Do not repeat a
rejected idea without new evidence.

## Measure useful work

The target is sustained 25 generated tokens/second while completing testing
tasks with low memory. Separate decode speed from prefill, time to first token,
thinking tokens/time, vision encoding, tool wait, and time to the next verified
action. Fast repetitive output or extra thinking can worsen task completion
despite a high token rate.

Aggregate decode as total generated tokens divided by total decode seconds.
Include individual cases so a mean cannot hide long-context regressions.
Never add overlapping GPU/CPU counters or move timing boundaries to claim a win.
Use actual production metrics and document their meaning.

Track current/peak service and foreground-app memory, system pressure, and
growth through the assigned long run. Distinguish physical footprint, resident
memory, reserved buffers, and OS file caching. Evaluate release after relevant
image, cancellation, New Chat, and unload changes. A single low sample cannot
prove bounded growth or absence of leaks. The upstream roughly 2 GB headline
does not guarantee memory at the owner's context and cache settings.

## Reproducible handoff

Keep commit and dirty-source identity, candidate diff, chip/RAM, macOS/Swift,
model identity, exact command/settings, exit, full timing footer, useful output,
memory, and deviations. Separate written, built, installed, and exercised.
A CLI result does not establish full navigation capability. An MCP success
does not prove a requested app state without evidence.

Build only when assigned and the shared slot is available. Use the existing
deployment script for assigned deployment. Do not enable experimental/profiling
controls, alter power settings, purge caches, duplicate models, or terminate
another run to improve numbers. Report unavailable checks and unmeasured
benefits plainly.
