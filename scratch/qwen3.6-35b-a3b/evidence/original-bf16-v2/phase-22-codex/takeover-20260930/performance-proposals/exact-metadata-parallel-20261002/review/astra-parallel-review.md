Scoped PASS: clear for Main to build and run this standalone metadata feasibility probe. No concrete execution blocker found. Production parallel validation is not approved by this review.

All 19 frozen-hashes.json entries match actual files. Reviewed receipt, candidate handle/measurement/command, and run.py. Key SHA256:
- forward.diff 120d84bb8e1754b5eed89b9deb8f7eec58682672c4d6791b3446f06c992dd3e1
- vs-metadata-cost.diff 6fb71e8cfc60a9b8ff5ca250db70a63f07cae27542e17c7790fff0a3658058a8
- OfficialSourceHandle.swift 2ae4425f4d178bdc226d10ed369ad2fc23becaea3a7f181b1075ffb8f601921a
- OfficialSourceMetadataCost.swift ac581c155e9f581124559f6e8e83b29fd740cafdbb8e3d20469a7068924040b9
- Command.swift 154ed243472bb818adc83396e7687eacf32c42c5d67f1f0f77decb2a285d8073
- run.py 897b1b5fd650ebccdec8c420490a6b3e218bba213eeb662371f0045a3ecd6541

Factory eligibility is established by a complete original serial validation on the same strongly retained handle and immutable receipt. This makes subsequent remember operations for the already accepted matching identities idempotent. The diagnostic object is not Sendable and the command calls it sequentially. The parallel helper is private and is not selected by production validation.

Every parallel entry retains original fresh openat flags, held fstat before short-circuit named fstatat, regular-file/nlink/full-fingerprint/held-name identity predicates, conditional locked remember, and deferred close before returning to its lane. Each fixed lane owns at most one entry FD. Contiguous partitions cover each of the 39 entries exactly once per scan. There is no arbitrary entry callback or dependency on a later job sharing a lane.

Lane counters are locally owned. A result lock publishes one completed snapshot/error, and caller merging occurs only after DispatchGroup.wait. Shared resource counters use short locks without filesystem operations inside them. The handle is retained through completion, protecting its directory FDs. Each lane continues after an entry error, drains all entries and closes its FDs. The caller chooses the lowest sorted index error only after all lanes finish and skips post-scan root on failure. Both independent scans preserve root/list-before, root-after-success and surrounding binding order.

The command checks successful primitive counts, final active resources zero and per-mode upper bounds. Worker1 maxima of one are structural constants from serial execution, not measured syscall tracing. Parallel tracked maxima describe receipt-file ownership, not every process or directory FD. Global queues may execute fewer simultaneous jobs than requested, so inspect actual peaks. Ignored close return behavior is inherited, not a new guarantee from a counter reaching zero.

Headline measurement includes allocations, dispatch, joins, resource locks, result merging and guards. Six forward/reverse cycles at 128 calls per mode yield 4608 clock-off calls. Separate clock-on controls total 192 calls. The <=0.85 gate is checked in both orders. Worker durations are overlapping sums, and held/named stats nest in fileCheckAndRemember. They must not be treated as elapsed partitions. Counter locks bias against parallel speed, rather than omitting parallel overhead.

Fixed sizeCheckTrustedReceipt initialization and metadata validation call no model, tensor or payload path. run.py preserves frozen/build/overlay/binary linkage, isolated flags, process/memory checks and owned-child watchdog. Its nominal 60-second timeout remains polled, with possible sampling overshoot. Primitive zero-byte claims remain explicitly static call-path evidence.

Receipt/output correctly preserve the three outstanding production differences: inter-entry observation ordering, absent caller-task cancellation inheritance in Dispatch, and extra simultaneous descriptors/resource-limit behavior. Current Task.checkCancellation inside workers is not a valid production cancellation bridge, but this synchronous standalone command has no task cancellation source. Lowest-index errors are deterministic, not identical to all serial temporal schedules. These limitations do not block the requested standalone timing question.

No build, test, model, metadata run or production edit was performed for this review. Clear for the frozen standalone comparison only.
