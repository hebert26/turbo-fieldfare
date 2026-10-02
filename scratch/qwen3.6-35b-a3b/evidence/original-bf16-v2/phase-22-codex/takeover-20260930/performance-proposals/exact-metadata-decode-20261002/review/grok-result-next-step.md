# Next step after the metadata pair

Do not change the gate. The four-lane metadata pair stays rejected. Prefetch stays rejected. Neither one is a path to 20 outputs/s.

OFF decode is 20.402703417 s. ON decode is 19.366123042 s. The ratio is 0.949193969. All 29 output IDs match. All 1120 route rows and 1120 plans match. Result minimum free memory is 31% OFF and 32% ON, and the audit memory checks pass. The same 2296 source checks take 3.285787829 s OFF and 2.605949390 s ON. The ON mean is 2605949390 / 2296 = 1.134995379 ms, above the fixed 1.050000 ms limit. Comparison exit 2 and the skipped reverse are correct. Source wall falls 0.679838439 s. Decode CPU rises from 26.394293 s to 29.598819 s. That CPU time includes more system time and is not added to the decode wall.

Twenty outputs/s for these 29 IDs is 1.450 s. This OFF decode is 14.1 times that budget. The later 25 forwards average 659.4 ms. Their miss slope is 1.119 ms per miss, with correlation 0.709 and an intercept of 449.4 ms. A forward with zero misses still sits near 449 ms on that fit, which is 2.2 outputs/s. The slope is an association across these 25 forwards. It is not proof that each miss is 1.119 ms of critical path.

Both arms load the same 5337 expert pairs and skip the same 3623. That is 33,577,500,672 logical bytes, 31.27 GiB, not NAND and not a unique working set. The rejected prefetch OFF arm of this same capture loaded those same bytes and recorded map wall 8.134569552 s, which is 3.84 GiB/s. This metadata pair does not record a map or a GPU union. A same-shape app capture put about 3.5 s of shader time inside about 6.7 s of command wait. Removing this map, this 3.286 s source wall, and that shader cost together still leaves several seconds. No hardware ceiling is established.

## One experiment

Teacher-forced layer-major replay of these 28 consumed tokens. Not a drafter, not the pair schedule, and not a larger permanent cache.

At each layer, route all 28 tokens with the existing CPU route. Load each distinct expert once into a transient buffer. Apply that buffer to every token that selected it. Release the buffer before the next layer. Keep the original per-token addition order and the volatile FP32 product boundary. Use the existing BF16 gate/up and down-add kernels.

`QwenMoE.submitGroupedExpertsBF16` already walks contributions and encodes each one. The comment at the loop says no mapping from an earlier chunk is reused. This experiment reuses one pair load across the tokens that selected that expert. It does not add a new reduction. Shader launches stay one per contribution, so the GPU union is outside this ceiling.

On these plans the 40 layers hold 2912 distinct experts (minimum 56, median 72, maximum 102). Serial decode loads 5337 pairs, so the most this can stop loading is 2425 pairs, 15,256,780,800 bytes (14.21 GiB). That is 45.4% of the miss bytes. At the 1.119 ms fit those 2425 misses are 2.71 s. At a proportional share of the 8.135 s sibling map they are 3.69 s. The decode then stays near 16–18 s, about 1.6–1.8 outputs/s. Windows of two tokens already fit in 16 slots (560/560, mean union 13.96) and lost on this request's acceptance histogram. Windows of four exceed 16 on 276/280 layers. Windows of eight exceed 16 on 120/120. Permanent 24-slot growth stays closed. The transient peak on this trace is one layer of 102 pairs, 612 MiB, then it is released. The 16 permanent slots stay 16.

## Measurement

Same 1175 prefill. The block arm then runs these 28 IDs and the serial arm produces the same 28. The EOS sample stays the production sampler's pending token. One runner. Free memory stays at or above 30%.

Keep every current check: both receipt scans, `validateBinding` before, between, and after them, fresh `open` and `openat`, the live marker read, live `entryNames` through EOF, the per-syscall before-read, after-read, and before-return checks, hit and publication `validateBoth`, mode, link count, and fingerprint. Membership scan stays off. Do not replace a fresh `openat` with a retained shard descriptor. Do not drop the post-map source check.

Reject on any output-ID, route-bit, or plan mismatch, or if free memory goes under 30%. Record logical pair loads and decode wall. The load cut is present only if loads are 3200 or fewer. It is absent if loads stay at or above 4800. If loads fall and decode wall falls by less than 2 s, the avoided reloads are not 2 s of critical path and this width stops. If wall falls by more than 4 s, stop and explain the extra before any flag. A wall still at or above 16 s agrees with the ceiling above. It is not 20 outputs/s, and it does not justify another lane count, a lower 1.05 ms mean, or a prefetch retry.

Main chooses. Sol does not code this from the note.
