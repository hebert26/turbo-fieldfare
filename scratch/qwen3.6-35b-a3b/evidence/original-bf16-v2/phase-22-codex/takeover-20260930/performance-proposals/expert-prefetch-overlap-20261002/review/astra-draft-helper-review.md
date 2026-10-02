Early live-draft review only. No final acceptance. Source changed during reading, including addition of publication-based import counting. Hashes below identify the final sampled files, not a frozen candidate.

1. Lifecycle race requires an enforced caller proof or helper correction. `start` stores `state.batch` before `launch` registers DispatchGroup entries. Concurrent `drain`/`cancelAndDrain` can observe zero entries, mark the epoch reusable and return while launch can still enqueue writers. `drain` also removes `state.batch` before awaiting settlement, leaving `cancelAndDrain` unable to discover/cancel/join that active batch. A Mutex around individual state accesses does not serialize this lifecycle. Account for launch before publishing the batch and keep a discoverable draining owner until settlement, or enforce and document a single serialized caller across these operations. Final runner integration must resolve this, especially cleanup during an awaiting drain. Sent directly to Sol.

2. `cancelAndDrain` says cancellation is expected, but `settled` wraps worker CancellationError values in QwenQuarantineReadFailure before checking the epoch flag. Pure cancellation during worker launch/read therefore escapes cleanup as a speculative read failure. If normal cancellation is meant to remain normal, classify an all-cancellation result accordingly while still retaining and propagating every non-cancellation source/read error in deterministic order. Do not simply suppress the aggregate.

No further ownership/bounds finding in the sampled protected owner/import path: admitted range and source references plus quota survive dispatched work; expert and byte bounds are checked before protected reads; original protected API is used. Import checks actual cache/source/range identity and known-none provenance, invalidates actual victims before writes, validates before copying, joins all copy/read streams, then preserves the original serial publication validation/cancellation/slot assignment sequence. Newly sampled counting marks imports at publication rather than first-stream copy. Full runner cleanup, root/source check placement and launch gates are outside this draft-helper verdict.

Capture counters sum worker wall durations and are not critical-path time. Final reporting must retain that distinction. No run, build, test, weight read or code edit performed.

Sampled hashes:
- `Runtime/Qwen/QwenProtectedEarlyPrefetch.swift`: `90fd83cbbf73f608300470bd6d21dcc19867bc8c56e723b73c111932a099eb71`
- `Runtime/Qwen/QwenEarlyPrefetchCapture.swift`: `31f298bd6f8e392e0cc9f25666090fa05295bf9eb5c71a398356f2472bd892cc`
- `Infrastructure/Streaming/PreadExpertStreamer.swift`: `0db7da4b3773cbd0bfa243823c379501d1681c01637b11295da14a186e9ed57f`
