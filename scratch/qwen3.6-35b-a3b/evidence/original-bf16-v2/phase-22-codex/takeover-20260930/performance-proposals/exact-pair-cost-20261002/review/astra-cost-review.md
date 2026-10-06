Scoped PASS: clear for Main’s bounded diagnostic run. No concrete blocker found in the frozen vs-pair timing delta. This is not speed qualification or acceptance of a full causal verifier.

Verified all 23 frozen-hashes.json entries against actual staged files. Relevant SHA256 values:
- patch/vs-pair.diff: 2b7a826524f01de38a0addd80680311ddbab77c7ca10a534a11a80fd87eb6571
- QwenExactBlockTypes.swift: 868f4616d3b5b31952c873d483dcf7ea0230aea5d1b446401a0368873473981f
- QwenExactBlockProbe.swift: 8c94cb9364398feb2f6e07395a30c909b37394f3a10263efe0617fbaded9a667
- QwenOfficialSourceRunner.swift: d975f261077504075319a81c9c93e6b4add0f48165ae2b31476cd7bb362ef478
- QwenMoE.swift: c3d54a7e8893499923e35822ecc502f962ee5fdba733cc5cfb5b51680255e628
- run.py: c891dc8c8d8dbcaf72a30f308d9ec8c9224d431b692f8214884c3e185588fbb7

fullSourceWallNanoseconds sums only twelve direct source-call counters, each wrapping the original synchronous model.revalidateSource once. Outer moe/sampler/turn/restoreReplay counters are excluded. Protected reuse and worker IO remain separate. Pair preGPU checks are nested in moe and must not be added to moe again.

Replay coverage is present: a separate collectsValues=false capture follows every accepted-prefix produce, merges its timing totals once, then the enclosing restoreReplay interval is recorded. This preserves captured block rows/routes without adding replay value copies. RestoreTurnBaseline has no direct source check to miss. BeginTurn and sampler checks inside arm are included. finishTurn outside arm is excluded, so this is not whole-request validation cost.

Successful-arm count cross-checks: serial four inputs contribute 328 direct source calls + one turn + four samples = 333. Pair four inputs contribute 330 + one turn + four samples = 335. First-row rejection with one sample and one replayed input contributes 330+1+1+82 = 414. These are new direct-source counters, not protected-read or legacy admission counts.

Arithmetic, work order, source-check positions, map/read/publication paths, settlement and rollback/replay operations remain unchanged. Replay capture adds the already-satisfied known-none eligibility guard. No new await, user callback or payload read is inserted between final admission and command encoding. Scalar updates stay on the existing sequential runner path without new locks or workers.

Interpretation limits: host elapsed counters include suspension where applicable. moeSubmit may overlap GPU execution and moeSettle measures host wait, neither measures GPU duration. Non-source stage counters generally close only on success, so failed runs do not provide complete partitions. Source wrappers close on failure. remainingWallUpperBound includes required protected IO and untimed work, not removable overhead. Timer-off controls disable scalar clocks but retain existing IO measurement and value capture. Single controls with evolving LFU state cannot statistically qualify timer overhead or speed.

run.py checks stage/build/overlay/binary identity before its guarded single child. Flags, bounded cache, memory threshold, process checks and owned-child watchdog remain as described. No builds, tests, model runs or source edits were performed. Clear to run this frozen diagnostic probe, then check counts before interpreting times.
