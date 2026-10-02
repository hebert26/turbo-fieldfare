Scoped PASS: clear for Main’s bounded metadata-only run after a successful pinned build. No concrete blocker found. No execution, build, test or model operation performed for this review.

All 15 frozen hashes match actual staged files. SHA256:
- forward.diff ac62ad054cca0b62a8b354c91f27eaeaf2aa9202e5a2e60f153d189d16d05dfb
- OfficialSourceHandle.swift bf14093470b912135bf33ab2bbc9d3999591480a3605f9df888db2dfc57a3150
- OfficialSourceMetadataCost.swift 65108e8656f50b037b357bc9dcd7e78e8af152b5037c5c2ca0ef9f8ff2f93b85
- Command.swift b659f5d63f6139c18d5bbfb9fc538d3eea8d0319fbe2bfc3c7dc5b2e5b684958
- run.py 3672b5b992adae2369928827ca9921d60bdf41abb40e079b4763b33e5db16d14

The actual validation delta preserves binding / receipt guard / scan / binding / scan / binding order. Scan retains root checks, listing to EOF, inventory subset check, sorted receipt iteration, per-entry cancellation, the exact openat flags, fstat-before-short-circuit-fstatat order, all regular/link/fingerprint/held-versus-named comparisons and conditional remember. Deferred close remains within each loop iteration and executes after success or a thrown file check. No descriptor retention, stat caching, bulk substitution or new observation policy is introduced.

Failure handling remains faithful: the wrappers defer counter closure when original checks throw. Listing catches only to record its duration and rethrows the same error. The file-open/stat failures already produce fixed replaced messages without consulting errno, so timing after the syscall does not discard an errno used by these guards. Inner binding/root helpers retain their existing error construction. close results remain ignored as before. No new suspension, callback, locking or concurrency appears.

Timing/count design is sound for successful validations: one total/receipt interval, three bindings, four root checks, two source listings, and 2*N open/heldStat/namedStat/fileCheckAndRemember/close occurrences. The command checks these counts against actual receipt N. heldStat and namedStat are children of fileCheckAndRemember, which must not be double-counted. Root duration includes its deferred named-root close. Total includes all instrumentation. Count-only OFF records calls with zero unmeasured durations. Failed checks may produce partial counts, but command reports failure rather than a successful measurement.

The command’s fixed sizeCheckTrustedReceipt branch reads bounded registration marker/receipt data and source metadata. Inspection of SourceTrustReceipt.verifyInternal confirms no fallback to fullSha256. No model loader, Metal context, tensor admission, source payload/header read or observer injection is called by the new command. Zero payload bytes is explicitly a static call-path assertion, not syscall tracing. Writing the result is the intended evidence artifact, not a source mutation.

run.py binds frozen stage, successful build, complete overlay and binary hashes before its existing pinned preflight and extra diagnostic process check. It sanitizes runtime flags and sets FAST=1/membership=0. The child is single, owned and guarded for memory and elapsed time. The nominal 60-second watchdog is polled and can overshoot by its sampling/wait duration; it is not a hard real-time limit. No old payload-reading metadata test is used.

Interpretation limits: the baseline uses the instrumented source with nil measurement, so it controls scalar instrumentation but is not an independent comparison against the old unwrapped binary. Six sequential short batches are sufficient for descriptive primitive attribution, not app-speed qualification. Wall counters are not syscall kernel CPU, process CPU covers slightly wider sampling boundaries, and filesystem cache state evolves. Keep failures and counts visible before drawing conclusions. Clear to run the frozen probe for cost attribution only.
