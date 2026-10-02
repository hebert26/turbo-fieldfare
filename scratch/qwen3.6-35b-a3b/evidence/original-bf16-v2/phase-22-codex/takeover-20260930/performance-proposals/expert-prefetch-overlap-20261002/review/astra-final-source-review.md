Scoped PASS: no source-level execution blocker found for Main’s bounded isolated comparison, conditional on successful build and wrapper/resource guards. This is not production acceptance or a speed claim. No build, test, model run or shard read performed.

Verified all 11 source-manifest entries. Manifest SHA256 `0b110b6afda98fe25397ff9ff28acc61d903d410f3eba5d567c4113a59da9d58`; forward patch SHA256 `d7728fc944f8f70a31a4dba0978509620ff9a8f15b79d6e6ee1faf0769c1990e`.

Both draft findings are resolved in the integrated call path. The read batch enters its group in init before publication, so join cannot race launch registration. A draining batch remains discoverable, its settlement group covers the drain owner, and epoch.importFinished covers the returned import lease. The coordinator awaits drain before acquiring cache locks and defers quarantine.finish until its dispatched fetch has settled. Runner failure awaits cancelAndDrain before restoring recurrent/KV state. Pure worker cancellations become CancellationError; mixed errors retain ordered source failures. Runner/probe cleanup combines production and speculative failures instead of hiding the latter.

Actual lifecycle is serialized by the runner operation: start returns synchronously, the next layer drains that one batch, import finishes before map returns, then the next batch starts. Task cancellation during drain cancels its epoch, joins the read work and settles it. Cancellation during actual fetch uses the existing cancellation flag and joins all copy/read streams before finish. Failure elsewhere in the forward reaches the outer drain. Existing top-of-operation guards do not encounter a prior pending batch after successful token completion. No helper lock is held while awaiting the workers. Strong source/range, quarantine-buffer and reservation owners survive those waits and every import job. The ordinary GPU lease retains its coordinator and actual cache quota independently.

Protected owner reads use the original admitted TensorRange API with checked expert/range bounds and exact destination sizes. At most one next-layer batch uses eight fixed pairs, serial pairs with two joined stream jobs each. A prediction only filters the next layer’s current resident inventory. It does not call cache.plan, mutate LFU counters, reserve or evict slots. Actual official routing, route weights, math and the sole actual demand plan remain unchanged.

Import verifies source/cache/range identity, layer/position epoch and genuine known-none hooks. It invalidates actual reserved victims before overwriting, retains fresh before-copy validation, joins every copy/read stream, then uses the ORIGINAL serial publication validateBoth as its after-copy check. Original cancellation/hook/check/slot assignment ordering and hit validation remain. No extra speculative source error is converted into a demand fallback. Every original full-source runner call is retained through the measured wrapper, including post-map validation before the same GPU submission.

OFF uses the existing wrapper around transaction hooks.none; ON uses static lower hooks.none. Both execute no user callback and knownNone_4 remains disabled, so this does not select a different worker-count executor. ON avoids the wrapper/withExtendedLifetime call overhead, which is a small comparison difference, not removed semantic work: the new typed coordinator quota owner preserves that lifetime. Explicitly supplied no-op callbacks remain ineligible through the private provenance witness. No arbitrary callback is admitted.

Both arms reserve, allocate and touch the same 48MiB quarantine through the existing quota and check free memory afterward. Quarantine never becomes a GPU resource or replaces an actual cache buffer. Memory feasibility is still a runtime condition, not established by review. The probe captures actual IDs/routes/weight bits/plans and requires drained idle state. Worker timing counters remain summed durations, not additive critical-path components. Wrappers/build receipts are separately owned and reviewed by Main.

Selected source hashes:
- `QwenProtectedEarlyPrefetch.swift`: `b51866b4896129ad863fedb090a5c13f2e6661a10394cfdea73a56c05e7c49d9`
- `PreadExpertStreamer.swift`: `ac0588532fc091f516e43482d5fd02af643ca82136a126f04bf6363e7c7c4a6a`
- `ModelExpertIO.swift`: `7aa3fb62555404812bcc87ceab221126d179f3d6221505472191ec0dbafe8d17`
- `QwenOfficialSourceRunner.swift`: `1f74d362d64e7df2f9085444ead2b2602d21c365edfa43b3508eaddd38050d73`
- `QwenExpertPrefetchOverlapProbe.swift`: `5bc4ed88189959dc069e75d8a3b49d3ad0f5a1a72289bd6a22ef33107cc565b2`

Recommendation: proceed only with the frozen isolated comparison and its fresh guards. No root integration is covered.
