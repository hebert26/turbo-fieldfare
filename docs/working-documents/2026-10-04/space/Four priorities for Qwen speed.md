# Four priorities for Qwen speed

Source: https://chatgpt.com/space/page_880397aba17881919107f5a8cb6818c0

Local snapshot: 4 October 2026. Source sequence: 25. Keep this copy uncommitted. The source Page remains authoritative.

**Index updated 3 October 2026. Point 4 is active. Points 1–3 are complete for the agreed scope.** Open a point below for its results, limits, decisions, and notes.

The goal remains sustained 20 tokens per second with correct output and enough memory for other apps. The latest accepted short app result is about 1.53 tokens per second. The target is not yet achieved. “Complete” below does not claim that every possible optimisation or every long-context workload has been proved.

## 1 Qwen context and compaction complete for agreed scope

[Open point 1 and add notes](https://chatgpt.com/space/page_1ee844c408c4819189ea686cc542c7b8)

Separate model settings are in place. Preserve Gemma’s choice and the existing recovery safeguards.

## 2 Expert memory and disk handling complete for agreed scope

[Open point 2 and add notes](https://chatgpt.com/space/page_72056a8fd1348191a7a9ce9597d5579b)

Bounded reads and reusable expert caches are in place. Previous rejected trials are recorded.

## 3 CPU and GPU work complete for agreed scope

[Open point 3 and add notes](https://chatgpt.com/space/page_71298161b6248191bf63419024bb461a)

Saved CPU/GPU changes reduced prompt-processing time in the measured app case.

## 4 Multi token prediction active

[Open point 4 and add notes](https://chatgpt.com/space/page_9f2c3451e6fc81919339e8da44a4d7f1)

Finish the exact whole-request comparison. Memory work is required for this point; it does not restart point 2.

[Qwen speed plan for the next 36 hours](https://chatgpt.com/space/page_401147ecde808191bdbac68ac69f54ea) — timed work, agent roles, pass rules, and fallback steps for point 4.

Keep detailed notes on the relevant child page. Keep this index short. Read the resume rules below after chat compaction. Your existing “Done” comments on points 1 and 2 remain on this index.

## Agent working agreement

| Agent | Work and reasoning |
| --- | --- |
| Main | Own scope, task dispatch, evidence acceptance, one guarded run at a time, and integration. Ensure reviewers do not review their own implementation. |
| Astra (xhigh) | Own deep investigation and innovation. |
| GPT-6 Luna (max) | Own documentation updates: status, results, decisions, proof limits, next step, and reasons not to repeat failed work. |
| Fable 5.1 (high) | Fable is expensive. Reserve it for important, difficult problems where its input matters. Use it for independent innovation and deep research alongside Astra. Do not use it for routine experiments. The existing Zed tab is available. |
| GPT-6.1 | Sol 6.1 in Codex owns implementation of selected solutions. Use `high` for bounded changes and `xhigh` for hard integration. |
| Opus 5.5 (xhigh) | Handle routine experiment analysis and review. Use the existing Zed tab. |
| Grok (xhigh) | Handle routine experiment analysis and review. Use the existing Zed tab. |

Use existing Zed tabs only. Do not use Luna for technical implementation or investigation; GPT-6 Luna (max) owns documentation. Use Astra for deep investigation. Reserve Fable 5.1 for important, difficult problems where its input matters; do not use Fable for routine experiments. Use Opus 5.5 and Grok for routine experiment analysis and review. Do not use every agent for each small task.

Look beyond the current code. Study primary documents, research, and reference source code. Cover Apple Silicon CPU and GPU work, shared memory, memory transfer rate, disk, work scheduling, power, and heat. Seek new speed gains that preserve exact model output and routing.

Each proposal must be bounded. State how it works, primary source evidence, measurements that separate possible causes, expected costs, the smallest experiment, and a stop rule.

The installed app and repository mentioned by Hebert are an investigation lead. Their exact identity is unconfirmed. Identify both before attributing methods. Then study how they assess the machine and use speculative or multi-token execution, where several possible tokens are checked together. Use primary documents and source code. Prior MLX references do not establish that MLX-LM is this app.

Fan noise or high GPU use alone does not prove a gain. Measure total request time, first-response delay, tokens per second, CPU and GPU work versus waits, memory use and transfer rate, power and heat where available, and app responsiveness. Keep all safety boundaries.

This agreement records roles only. It does not start research or experiments.

## Saved Page instructions

Resume rule requested by Hebert on 3 October 2026: points 1–3 are closed for their agreed scope. Point 4 is the active priority. Read this Page and the newest saved code/run evidence before choosing work after chat compaction or handoff.

If a restored summary says to begin points 1–3 again, treat that as a possible lost-context error. Recover the existing results first. Do not silently restart old investigations or describe rejected trials as new ideas.

Only a new request from Hebert or a concrete measured regression or dependency can justify a bounded return. State the failed requirement, cite the prior result, explain what changed, and define the check that will finish the work. Keep required memory work under point 4 and return to its multi-token comparison when that dependency is resolved.

Do not turn a closed workstream into a claim that all possible workloads have passed. Keep proof limits, failed runs, and experimental status visible. Update each point with what was done, what worked, what was rejected, why it stays closed, and the exact next action. Remove stale “start point 1” notes. Do not create timers.

Follow the Agent working agreement above. GPT-6 Luna maintains the documents at maximum reasoning. Astra focuses on deep research and innovation. Main retains scope, dispatch, evidence acceptance, one guarded run at a time, and integration. Verify Fable 5.1 access and its available reasoning setting before use. Do not silently substitute another model.

Keep the existing run safety rules. Do not create timers or unit-test code during experiments. Commit stable milestones. This agreement alone does not start research or experiments.

