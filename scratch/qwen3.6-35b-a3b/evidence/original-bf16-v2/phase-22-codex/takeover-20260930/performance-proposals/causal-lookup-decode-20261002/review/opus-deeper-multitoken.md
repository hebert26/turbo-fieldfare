Opus deeper multi-token design, 2026-10-02 (read-only at ab21aea; causal-lookup decision and receipts, the exact-pair-probe and exact-pair-cost candidates and reviews, live runner, KV and linear-state code, pinned official config, index and README; no builds, runs or weight reads). Runner line numbers refer to exact-pair-cost-20261002/candidate; live files are cited by name.

## 0. Facts this rests on

**The causal run gave no speed result (measured).** Free memory went 46% at launch → 41 → 35 → 31 → 30 → 29% within 16 s, during load and prefill of the OFF child (serial, no new feature). The guard stopped it, so the experiment consumed about 17 points of free memory before generating anything. Any design must therefore be memory-neutral, and the next run needs launch free ≥ about 48%.

**Real-app cost per forward (measured, known-none-off app-001, 31 forwards, 676 ms each):**

| Item | ms per forward |
|---|---|
| Map | 297: 204 misses × 1.26 ms, 116 hits × 0.26 ms, intercept |
| GPU busy | 113: MoE 56, linear 38, full attention 11, head 6.7, router 1.1 |
| Submit to GPU start | 90 |
| GPU end to resume | 15 |
| CPU attention | 25 |
| Rest | 138, of which full scans ≈ 97 (82 × 1.19 ms) |

**Shareable versus per-position cost (estimate).** A verify block of K rows can share only:
- per-command launch costs (about 105);
- dense weight reads (about 45);
- the map intercept;
- some CPU work.

That is S ≈ 190 ms. Everything else is paid again for each verified position, N ≈ 486 ms:
- **Misses:** each token's new experts are still read once. LFU16 already keeps the previous token's 8 experts, so union saves little: warm 434–435 misses against 446.
- **Per-token full scans**, required by contract-decision.md.
- **MoE GPU:** the routed experts differ between tokens.
- **CPU attention.**

**Consequence (estimate).** The block ratio is r(K) = N/T + S/(K·T) ≈ 0.72 + 0.28/K, where T is the 676 ms serial forward. With any proposer and perfect acceptance, multi-token is capped at T/N ≈ 1.4×, about 2.1 tok/s.
- Native MTP with 1 draft needs acceptance p ≥ 0.76 just to break even, and gains at most about 13% at p = 1.
- Lookup with 3 drafts needs at least 2.1 accepted per proposing cycle. The tool screen accepted 13 of 27 drafts.
- Twenty tok/s needs about 50 ms per position, roughly 10× below N.

This bound depends on the measured per-miss and per-check costs and the current contract. It is not a hardware claim. Point 4 multiplies whatever per-position cost exists; it cannot replace reducing that cost.

**Three different mechanisms:**
1. **Prompt processing** (111 ms per token: 1,175 tokens in 131 s) is fast because:
   - hundreds of known tokens share each expert read;
   - prefill validates once per group (candidate runner :531, :767, :780);
   - linear layers batch 16 tokens (:586).

   Small causal blocks get only the linear batching. Per-group validation for verify rows is Main's and Hebert's contract call. With it, N falls by about 97 and the ceiling rises to about 1.75×. It is not assumed here.
2. **Prompt lookup:** weightless, causal suffix matches. It proposes nothing on non-repeating text (7 of 16 tool cycles).
3. **Native MTP (pinned metadata):**
   - The source has `mtp_num_hidden_layers 1` and `mtp_use_dedicated_embeddings false`, with 19 `mtp.*` tensors (shards 25–26): `fc` (2H→H), `pre_fc_norm_embedding` and `pre_fc_norm_hidden`, one full-attention layer (q/k/v/o, q/k norms), a full 256-expert MoE with shared expert and gate, and `norm`.
   - It uses the main embedding and `lm_head`.
   - The README recommends vLLM `qwen3_next_mtp` with 2 speculative tokens, or SGLang NEXTN with 3 steps and 4 draft tokens.
   - The pinned transformers code ignores `^mtp.*`, so no forward code is pinned locally. The vLLM source must be pinned before implementation: the order of the embedding and hidden concatenation, and the pre-final-norm hidden input.
   - Draft arithmetic never affects output exactness, only acceptance. Its source reads still need every check.

## 1. Chosen design: one exact verifier and rollback core for any proposer

Recurrent-only rollback with batched shared rows. This core serves both lookup and MTP. It is the only part that decides whether point 4 can pay, and it must exist before any MTP work.

**A. Rollback by recomputing only the recurrent state.** This replaces both baseline-plus-full-replay and retained per-boundary snapshots.
- Full-attention KV is append-only. Rows below base+keep, written by the block, are already proven bit-identical to serial. The only non-append-only state is the 30 linear layers' convolution and recurrent state.
- During the block, save each linear layer's normalized input rows (runner :582–616, `normalizedRows`). That is K × 30 × 2,048 floats, 0.94 MiB at K = 4.
- `commitAcceptedPrefix(keep)`, new, beside `restoreTurnBaselineCore` (:932–950):
  1. Restore the linear turn baseline with the O(1) retained checkpoint.
  2. For each linear layer, re-append the saved rows 0..<keep through the serial `append(tokenCount: 1)` path (:626–628) with the same `useGPULinearPreparation`. Discard the outputs. No expert reads, attention, head or MoE run.
  3. Truncate full KV to base+keep.
  4. Set `committedPosition`, refresh the turn snapshot, then run a fresh `revalidateSource` before `finishTurn`.
- **Cost (estimate):** about 30 linear commands per kept row, roughly 40–60 ms, against keep × about 676 ms for today's replay (QwenExactBlockProbe.swift:242–256).

**B. Batch the rows' shared work** (exact pair schedule unchanged):
- K-row router per layer (today it runs per row at :641–643);
- K-row head (per row at :841–859, which reads the 1 GB `lm_head` K times);
- per-token checks, pair maps and fused pair MoE commands stay exactly as reviewed.
- Allow K ∈ {2, 4} at the eligibility guard (:439).
- Full-attention K-row projections are deferred: CPU attention is per row anyway.

**C. Turn baseline per cycle.** `beginTurn` (:908–923) makes a 64 MiB CPU `clone()` of the linear state every cycle, and restore writes it back. Use `linearState.retainCheckpoint()` instead. It is O(1) and copy-on-write (QwenLinearAttentionState.swift:292/:310), the same mechanism `produceToken` uses at :328. This removes a per-cycle copy and a transient 64 MiB at the guard.

**Files** (new candidate copied from exact-pair-probe; production untouched while `exactBlock == nil`):
- `Runtime/Qwen/QwenOfficialSourceRunner.swift`: A, B, C.
- `Runtime/Qwen/QwenFullAttentionKV.swift`: `prefixSnapshot(position:)`, same owner and lineage, all full layers at p and p not above any committed position. Restore via the existing `restore` (:417), which raises lineage when it truncates.
- `Runtime/Qwen/QwenExactBlockTypes.swift`: per-cycle proposal, verify, commit and recovery timers.
- `Runtime/Qwen/QwenExactBlockProbe.swift` and `TurboFieldfareCausalLookupDecode/Command.swift`: call the commit, and add a K variant.
- `QwenBF16Weights.swift` only if `encodeProjection` K-row for the router and head shapes needs a guard.

**Correctness and lifetime rules:**
- Saved rows belong to one block and are cleared on commit, rollback or error.
- The commit is all-or-nothing. If any re-append, truncate or check fails, restore the full turn baseline and use the existing serial replay, or mark the runner unusable as today. Linear and KV must never disagree on position.
- No await between KV truncation and `finishTurn`.
- Batched router and head rows must match per-row output bit for bit. The probe's raw-row and route hashes are the gate. On mismatch, keep per-row and ship A and C alone.
- Eligibility stays `hooks.isKnownNone`. Every per-token check is kept. Only the duplicate full re-execution, with its own checks, disappears.
- A sampled EOS stays pending, and rows after it are dropped, as now.
- The full-conversation lineage case remains an integration requirement.

**Memory and time budget:**
- **ON memory:** about 1 MiB of saved rows and about 4 MB of K-row router and head buffers. It removes a 64 MiB per-cycle clone. Net ≤ +5 MiB.
- **MTP later:** about 155–205 MiB (dense about 75, 8–16 expert slots of 48–96, KV about 32 at 8K). Shapes are inferred from the config, and Sol must check them against headers. MTP would need an offset first, for example per-token protected reads instead of the resident 1.017 GB BF16 embedding table (QwenBF16Weights.swift:118/:222).
- **Time (estimate):**
  - rollback falls from keep × about 676 ms to about 50 ms;
  - batching saves about 25–35 ms per extra row;
  - the expected causal r(4) is 0.75–0.85. The one cold observation is 2.03 s against about 2.7 s serial, about 0.75.

## 2. Decisive experiment

It counts proposal, verification and recovery, with no oracle and no tuning.
- **Arms:** fresh children, same prompt and flags.
  - OFF: serial.
  - ON-K4: fixed lookup with at most 3 drafts (the frozen policy).
  - ON-K2: fixed lookup with at most 1 draft. This measures r(2), the bound on MTP-1, without building MTP.
- **Start rule:** start only when launch free is ≥ 48%. Abort at 29% as now.
- **Report for each cycle:**
  - proposal ns;
  - block wall, with commit and rollback wall separately;
  - sampled and accepted counts;
  - misses and scans;
  - memory.
- **Causal r(K):** outputs are exact, so OFF's forward walls at the same positions are the serial cost of the same inputs in the same history. r(K) = block wall ÷ Σ OFF forward walls at those positions, over full-accept cycles. Report the total decode ratio too.
- **Cost:** about 3 minutes per child. The first order takes about 9–10 minutes. Run the reverse order only if an ON arm is ≤ 0.95.

**Stop rules, fixed before the run:**
- Any route, raw-row or state mismatch, or a commit or rollback state mismatch at any keep: fail. Fall back to retained-boundary snapshots only if memory allows.
- Commit or rollback wall per rejected cycle above 70 ms: A fails.
- r(2) ≥ 0.88: MTP-1 can gain at most about 10% even at p = 1. Do not build MTP.
- r(2) ≤ 0.80: stage MTP-1 with draft-in-verify, after the memory offset and pinning the vLLM reference.
- ON-K4 at or below 0.95 in both orders: plan lookup integration for structured output. Otherwise lookup stays off.
- Both r values ≥ 0.85: report to Hebert that point 4 is capped near 1.1–1.2× until per-position cost falls, and stop point 4.

**Prediction, recorded before the run:** r(4) 0.75–0.85, r(2) 0.85–0.9, ON-K4 about 1.0–1.2× OFF on this request.

## 3. Alternatives and what would overturn the choice

- **MTP first.** Under N it gains at most +13%. It adds about 155–205 MiB at the guard, a new layer, MTP KV and an MTP prefill pass, and its acceptance is unmeasured. It wins only if the measured r(2) is ≤ 0.8, or if contract changes cut N.
- **Retained boundary snapshots.** They need 63–190 MiB. Recurrent-only re-append needs about 1 MiB.
- **Progressive 2+2.** Covered by the K2 arm.
- **Overlapping one row group's reads with another's compute.** The prefetch run measured scans ×1.94 and other compute +15% under concurrent reads and checks.
- **Larger blocks through transient union buffers.** Union saves little against LFU16 and costs 240 MiB or more.

**The choice is wrong if:**
- causal r(4) ≤ 0.65: much more is shared than modelled, so go MTP-first;
- recurrent re-append is not bit-exact;
- ON lowers minimum free by more than 1 point against OFF.

Decision: build the proposer-independent exact core first (recurrent-only rollback, K-row router and head, O(1) turn baseline). Measure causal r(2) and r(4) with fixed lookup. That measurement alone decides whether native MTP or lookup can pay on this runtime.
