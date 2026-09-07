# Gemma architecture and expert residency

Read for routing, expert caches, disk I/O, or prediction. Implementation facts
were checked 2026-09-07. Proposed improvements below remain hypotheses.

## What chooses experts

Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift defines the
pinned Gemma 4 26B-A4B shapes: 30 layers, 128 routed experts per layer, top-8
selection, and a separate shared expert. Twenty-five attention layers use a
sliding window and five use full attention. Check ArchConfig and the manifest.

The router selects experts from each token's current hidden representation at
each layer. It does not route a whole user request once to a fixed expert set.
Later layers depend on earlier computation, and later generated tokens depend
on earlier tokens. Expert IDs are layer-local. Expert 7 in two different
layers does not identify the same weights.

Keep common weights mapped and stream selected routed weights into reusable
slots. Trace Runtime/Inference/ModelExpertIO.swift and
Infrastructure/Streaming/PreadExpertStreamer.swift under Sources/TurboFieldfare/.

## Existing caching

makeExpertCachePlan matches requests to occupied slots, preserves hits, assigns
slots for misses, and protects excluded slots. executeExpertCachePlan reads
only misses and publishes their IDs after successful reads. The default is LFU
with recency as a tie-break. LRU also exists. Trace ownership across planning,
reading, and GPU use before altering concurrency.

The owner's 32-slot setting applies per opened layer, not to one global pool.
Estimate capacity from actual aligned allocations across all opened streamers.
More slots may reduce repeated reads but cost memory. A cached expert still
requires computation and may not be selected again. Measure useful reuse and
the actual wait on the critical path.

## Prediction and retention experiments

Retaining frequently needed experts can help when misses delay inference.
The cache already does this. Further gains from better bounded retention or
speculative prefetch must be demonstrated.

For an assigned experiment, examine per-layer reuse and the current policy.
Prefer cheap existing signals such as recent selections and hit/miss counts.
An expensive predictor or unbounded routing log needs evidence that saved
waiting exceeds its cost. Do not assume experts have human-readable specialties
such as profile creation or bookmark testing.

Predictions may change when weights become available. The actual router must
still determine which experts execute with their unchanged weights and mixing
coefficients. Wrong predictions must fall back to fetching the true choices.
Never favor cached experts by altering routing, skip required experts, or
hardcode expert lists for an application.

Bound speculative reads and protect active slots. Compare misses saved, bytes
read, I/O wait, evictions, wasted prefetch, CPU/GPU overhead, and total physical
memory. False predictions can evict useful weights, compete for SSD bandwidth,
and worsen pressure. Prefill's known routed pairs within a processed chunk
differ from predicting unknown future decode tokens.

Read apple-silicon.md before changing overlap or slot lifetimes, and
performance-evidence.md for assigned comparisons. Preserve correctness and
reject speedups whose memory or long-context costs exceed the assigned budget.
