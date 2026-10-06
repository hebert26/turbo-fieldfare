Astra baseline memory probe review — sampled source, final freeze pending

No concrete blocker found in the measurement delta against recurrent-prefix-recovery-20261002. Final source manifest recheck remains required. Main owns wrapper admission and execution.

The recorder emits fixed schema numeric fields and a bounded request identity. Each JSON line is under4096 bytes, written directly to stderr with short-write and EINTR handling. Missing/failed records remain absent. There are about50 phase lines on this fixed serial path, not per-token/payload logs. The wrapper redirects stderr to a regular file, avoiding a pipe-reader dependency. Direct write bypasses userspace stdio buffering and survives ordinary child termination once accepted by the kernel; it does not establish fsync/power-loss durability. A killed child can leave a partial final line, which must not be parsed as a complete event.

All inference/source checks, flags, buffer geometry, prompt inputs and target arithmetic remain unchanged. The probe adds sparse synchronous encoding/writes and actor calls at already-idle boundaries, so it is resource attribution with some overhead, not a clean performance comparison. Model load reads are unchanged. No model or payload read is added by the recorder.

ModelExpertIO register already owns its residency lock. The new private phaseSnapshot reads fields under that existing lock without reacquiring it. record has no callback, actor hop or owner retention, so no lock cycle is introduced. It does write two bounded lines while holding the lock; the fixed regular-file sink is important to that assessment. requestResidency/commit/queue attachment order is unchanged. The after event means the residency REQUEST returned, not that physical residency has completed. Runner layer-boundary record obtains a normal locked snapshot outside register. No MTLBuffer or command is retained by recorded JSON.

currentAllocatedSizeBytes is explicitly Metal-device allocated bytes. residencyAllocatedSizeBytes is the set's allocation accounting. Neither is named physical footprint or RSS; neither proves pages are resident. Snapshot fields before set.commit can reflect pending-set accounting and must not be treated as independently verified physical bytes.

Clock compatibility: DispatchTime.uptimeNanoseconds is converted mach_absolute_time on Darwin, per [Swift Dispatch source](https://github.com/swiftlang/swift-corelibs-libdispatch/blob/main/src/swift/Time.swift). [Apple documents CLOCK_UPTIME_RAW nanoseconds as the equivalent clock](https://developer.apple.com/documentation/kernel/1462446-mach_absolute_time). sampler.py resolves CLOCK_UPTIME_RAW=8 from the installed SDK, rather than assuming a Linux clock ID. Thus phase/sample clocks share the same uptime domain. Sampling intervals and marker placement still bound attribution; timestamp capture precedes the Metal property read and stderr write, so do not infer nanosecond-perfect simultaneity or align by UTC string.

Sampled hashes
- Runtime/Qwen/QwenBaselineMemoryPhaseProbe.swift: 4dcb19aaa955b1050517724555b83f7499c225d241946efdc12b4263aa806b5b
- Runtime/Qwen/QwenOfficialSourceRunner.swift: 2f8b94ffbc4ed4a40ede3f857979ba5215822519f862c7d99b82c4320e37c136
- Runtime/Qwen/QwenExactBlockProbe.swift: fb5a777e554e0fd68bcf0ac2ad9821b1f12ffcb00fb1865a2e3ec8a08a08ec8b
- Runtime/Inference/ModelExpertIO.swift: 9e6a773b11f888b0a136b821de71621f3583ec85b9531c34e914d72c5a341ae8

Next action: match these actual files to Sol's frozen inventory before issuing the final scoped verdict. No tests, builds, model runs, payload reads or source edits were performed.
