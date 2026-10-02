# PASS — scoped source admission and failure settlement

Reviewed actual frozen pair delta against the rejected probe and the previously reviewed v2 admission. All candidate entries in `frozen-hashes.json` match their files. The actual delta filename is `patch/vs-rejected.diff`, SHA256 `a5ce52c18e5ba95e3fd3c494910ee0ca3f7de3341325d74201ffcd205995a337`. No concrete blocker found in this scope. No builds, tests, model runs, shard reads or source edits performed.

- The runner admits only known-none, four-token, top-eight, 16-slot diagnostic execution. Pair membership is fixed to 0/1 then 2/3. Each pair receives two separate fresh full-source pre-map checks. Mapping remains the original protected physical map; its cache/read/publication implementation hashes match the reviewed v2 files.
- The actual submission work is constrained to one complete pair, exactly 16 contributions, every original route rank once per token, at most 16 leased experts and the correct clear flag. V2 admission receives this actual work and requires identity with the submitted lease. It derives both touched tokens from that work, performs protected checks on every additional logical expert use, and then independently revalidates the full source for each token.
- All added row allocations and binding checks precede admission. Admission still finishes before `lease.submit` creates/encodes its command. There is no await, arbitrary callback or source payload read between the final checks and encoding/commit. Authentication failure follows the no-command cancellation path, so it cannot commit partial GPU work.
- Zero separate shared checks is correct here. Committed `3fe3305` already authenticates each token/layer's routed and shared work together in one lease-owned command. The pair preserves each token's pre-map and post-map checks; it does not replace the two consumers' checks with one. The rejected probe's additional shared check existed only because shared work was a later command.
- Shared row buffers are appended to `retainedBuffers` before submission. Both normal completion and partial-encoding failure call the existing settlement path with those buffers and the lease. That path waits for command completion handlers, retaining buffers and pins. Cancellation marks/discards the lease but does not release GPU-owned state early. Pair B is mapped only after pair A's awaited submission/settlement returns successfully. A failed A cannot continue to B or CPU residual readback.
- The source-check gates require 160 pre-map and 160 pre-GPU validations, four token/final checks, 80 maps/commands, zero separate shared commands/checks and mapped first uses plus reuse checks equal to 1,280. These are useful completeness assertions; runtime results remain unverified.

Reviewed candidate hashes:

- `QwenMoE.swift`: `bc01449b0db5af3044a401251e38da7cc30bfa7f704b0ad5516503c41db5c10b`
- `QwenOfficialSourceRunner.swift`: `82006005a9c1bcd550fefeac70fcc57c16735851f3117d3bb5746c5fc6997d51`
- `QwenExactBlockTypes.swift`: `883250021f4dbfe893922f5875d5374f6799f6fe066e3319944a0beff28633d0`
- `ModelExpertIO.swift`: `6edae3cec52c39dfd5a5698c821ee4d304e2fc4537c0c1e74289c43e2f2dcf09`
- `PreadExpertStreamer.swift`: `ec7ca99260123597e60ba860493b8363fdf3a5e4e04be69157a7d8c1e030bec1`

Scoped PASS for admission and settlement only. This is not full arithmetic/harness review, compilation verification or performance acceptance.
