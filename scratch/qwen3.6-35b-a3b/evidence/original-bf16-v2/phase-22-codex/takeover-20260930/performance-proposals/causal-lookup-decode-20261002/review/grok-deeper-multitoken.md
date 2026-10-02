# Deeper multi-token design

Use the existing 16-slot grouped chunk scheduler for every causal window of two or more inputs. Stop using the fixed exact-pair kernel. This is the strongest practical change toward 20 tokens/s that stays inside the current artifact, the 16-slot cache, and the 30% free-memory guard. It is not a 20 tokens/s claim. Sol codes it only after Main chooses it. Do not retry the aborted run now.

## What the aborted run is

`causal-lookup-decode-20261002/decision-001.json` is `RESOURCE_GUARD_ABORT_NO_SPEED_RESULT`. HEAD `ab21aeadac0297ab253b56b0b8aab0fd6ebb5e6b`. The OFF child lived 16.025 s, launch free 46%, minimum free 29%, child exit -15, `completedGeneration` false, `speedResult` null, ON never started. `memory.jsonl` free percentages are 46, 41, 35, 31, 31, 30, 31, 30, 29. The guard stays 30%. This machine state cannot host another child until a later quiet launch stays at or above 30% through load.

## Three different mechanisms

Prompt processing is the normal grouped prefill. `QwenOfficialSourceRunner.swift` builds one work list per layer, then maps unique experts in groups of `expertSlotCount` (16). The first map loads the group. A later chunk of that same group maps the subset as hits and does not evict the group (`:728-743`). Many known tokens share each load. That path is available only when the tokens are already known.

Prompt lookup is `probeProposal` (`QwenExactBlockProbe.swift:31`). It reads `history + [pending]` and copies up to three following tokens from an earlier match. On the frozen tool screen that accepts 13 of 27 drafts, leaves 7 of 16 cycles empty, and checks 43 positions to commit 29 tokens. The ON driver calls it only after the real sampler produces `pending` (`:153-154`). Saved future IDs are not an input. A lookup window is at most four inputs, and only a four-input window uses the grouped kernel (`:169-175`). Shorter drafts run serially.

Native MTP is one extra layer. `QwenOfficialIdentity` pins `mtpNumHiddenLayers == 1`, `numExperts == 256`, and `numHiddenLayers == 40`. `OfficialTensorMap` and `GTurboFormatV2.qwenIgnoredTensorNames` list the same 19 `mtp.*` tensors, including `mtp.layers.0.mlp.experts.down_proj` and `gate_up_proj`. The manifest feature is `qwenMTPExcluded`. Those tensors are not in this gturbo. The text-expert pair measured elsewhere is 6,291,456 bytes, so the omitted MTP routed experts are 256 × 6,291,456 = 1,610,612,736 bytes (1.50 GiB) before the attention and shared-expert tensors. An MTP draft still needs a main-model verification of every accepted token. Acceptance is unmeasured. Loading the 19 tensors is not the first change, and it fights the guard that just fired.

## The change

Today, any `exactBlock` prefill ignores the chunk scheduler and walks fixed pairs `(0,1)` then `(2,3)` (`QwenOfficialSourceRunner.swift:680-693`). Each pair builds its own expert set and its own lease. `QwenMoE.submitGroupedExpertsBF16` then refuses the command unless `tokenCount == 4`, the tokens are exactly `[0,1]` or `[2,3]`, `work.count == 16`, and the lease holds at most 16 experts (`QwenMoE.swift:1151-1158`). An expert selected by token 0 and token 2 is therefore mapped twice. Inside one command the kernel already applies every work item on that command's lease (`:1233-1238`). `QwenExactBlockConsumerAdmission.validate` already treats a repeated expert in the same work list as one mapped use plus an independent hit-range check, and it revalidates the source once per logical token (`QwenExactBlockTypes.swift:36-51`). The pair split is what keeps that sharing from seeing the whole window.

Code, after Main chooses:

1. `QwenOfficialSourceRunner.swift` exact-block branch (`:680-727`). For a causal window of two or more inputs, use the same 16-expert group loop as `:728-770`. Pass every contribution for the experts in the group, not a two-token filter. Keep the per-token `revalidateSource` that loop already performs. `initializeOutput` is true only on the first chunk of the layer.
2. `QwenMoE.swift` exact-pair guard (`:1151-1184`). Accept a chunk whose expert count is at most 16 and whose token indexes are any subset of the current window. Shared-expert rows stay one private buffer pair per referenced token, command-local. Remove the requirement that the tokens be `[0,1]` or `[2,3]` and that `work.count == 16`.
3. `QwenExactBlockProbe.swift` (`:169-175`). Use that grouped prefill for every ON window with at least two inputs. Keep serial `produce` for a one-token cycle. Do not edit `probeProposal`, the suffix widths, the 16-slot count, or the MTP omission set.

## Correctness and lifetime

The serial arm and the chunk arm receive the same causal inputs, including drafts the sampler will reject. Raw logit rows, route IDs, route weight bits, and the consumed-prefix state must match before a wall time is trusted. The layer plan check stays: each token and rank appears once (`QwenOfficialSourceRunner.swift:662-677`). Each token that touches a command keeps its own full-source check. Do not collapse those checks into one check per chunk.

`pending` comes from the sampler. The correction sampled at the first mismatch stays pending and is not fed back as a draft. Recovery stays the current one: if `keep < inputs.count`, `restoreTurnBaseline` and serial-`produce` only `0..<keep` (`QwenExactBlockProbe.swift:190-196`). Release the chunk lease and the private row buffers before that restore. `committedPosition` advances only through the kept prefix. Do not add prefix snapshots, a second cache, or extra slots. A sampled EOS stops before that EOS is consumed, as the current driver does.

## Memory and time budget

Permanent routed cache remains 16 slots. The added live memory is the command-local shared rows. Four tokens at hidden size 2048 and two Float rows are 64 KiB, freed with the command. No MTP residency. The child stops if free memory goes below 30%. ON does not start unless OFF has completed a generation at or above 30%.

20 tokens/s is 50 ms per committed token. The warm four-input pair median is 1.562705 s against serial 1.654878 s, ratio 0.944303 (`exact-pair-cost-20261002/decision-001.json`). That is 391 ms per input even when all four are accepted. Chunk sharing can remove a second lease for an expert used by tokens in different pairs. It does not divide the mixer, the head, or the per-token source checks by the window length. Recorded prediction: a full-accept window moves by less than 2× versus that pair block. This design does not predict 50 ms.

## Decisive experiment

One quiet child at a time. Same flags as the aborted receipt: validation-fast, residency, GPU linear preparation, and grouped linear prefill are `1`; projection batch, membership scan, known-none-4, and token capture are `0`. Prefill of the 1175 prompt IDs stays outside the ratio. OFF is serial produce. ON is the frozen proposer plus chunk verification plus the existing recovery. Report proposal time, verification time, and recovery time on every cycle, and use their sum as the decode wall. Also report misses and logical payload.

Pass only when output IDs, the accepted route hash, the accepted raw rows, and the final state match, free memory stays at or above 30% for the whole child, and the ON wall is strictly below the OFF wall. Stop this line if the ON/OFF ratio is at or above 1, free memory goes below 30%, or the time per committed token stays above 200 ms. Do not load MTP and do not grow the cache after that stop.

## Alternatives, and what would disprove this choice

Progressive pair stop keeps the pair kernel and skips pair `(2,3)` after the first draft mismatches. It avoids some suffix work. It still maps a shared expert once per pair, and the earlier screen price stays above serial. Choose it instead if the chunk arm does not lower payload against a pair arm on the same window.

In-place prefix commit removes the serial replay. The measured keep-1 replay is 0.974712 s inside a 4.521574 s rejection, one observation. Replay removal leaves the 0.944303 verification ratio, and retained boundaries add memory on a machine that just crossed 29%. It is not this change.

Native MTP is the proposer to measure after this verifier, not before. It becomes the better spend only if a later artifact can hold the 19 tensors, a quiet OFF run stays at or above 30% free, and an MTP-plus-chunk wall beats chunk-only by more than the MTP layer's own time. A chunk result at or above the current pair wall disproves the sharing claim even if the wall moves for another reason. A chunk result under 100 ms per committed token is the evidence that would reopen a path toward 20 tokens/s. This note does not expect that result.
