# Omnigent technical task prompt template

Choose the team for each task before sending the prompt. Replace every `<placeholder>` and remove unused rows. This file does not start work or change agent settings.

## Available agents

Read from `/Users/dev-machine/.omnigent/agents/turbo-main/agents/` on 27 September 2026. Configuration may change, so confirm availability in the receiving session before dispatch.

| Agent | Configured model | Allowed role |
|---|---|---|
| `turbo-mac-implementer` | GPT-6 Sol | Production implementation. Does not edit tests or fixtures. |
| `grok-implementer` | Grok 4.6 | Production implementation. Does not edit tests or fixtures. |
| `turbo-luna-test-engineer` | GPT-6 Luna | Unit tests and test fixtures, plus focused test execution. Does not edit production code. |
| `turbo-mac-review` | GPT-5.6 Sol | Read-only implementation review and advice to Main. No edits, builds or test execution. |
| `independent-reviewer` | Claude Opus 5.5 | Read-only implementation review and advice to Main. No edits, builds or test execution. |

Main coordinates the work, integrates it and makes the final decision. A review result is advice to Main, not permission for the reviewer to implement changes.

## Choose the team

- A starting option to evaluate is `turbo-mac-implementer` for coding, `turbo-luna-test-engineer` for tests and `turbo-mac-review` for review. This is not an automatic assignment or a price comparison.
- Select `grok-implementer` only when Hebert chooses to spend its limited credits on this task. It can replace the other implementer or own a separate coding piece.
- Select `independent-reviewer` only when Hebert chooses its higher cost for this task. Normally select one reviewer, not both.
- Use two implementers only when there are useful, separate coding pieces. Name each agent's files and agree the interface before parallel edits.
- Swap agents within compatible roles. Never give either reviewer unit-test writing, and never give the test engineer production edits.
- If the selected roster cannot do the job, report the exact mismatch. Do not change permissions, models or the selected team silently.

## Copy and fill this prompt

```text
Hebert authorizes ONLY <phase/task ID and exact title> in:
/Users/dev-machine/dev/turbo-fieldfare-personal

OUTCOME AND SCOPE
Implement <one observable behavior> in <exact files/module> because <why it matters>.
Successful result: <what the user or caller can observe>.
Current starting point: <relevant existing behavior and accepted dependencies>.
Read <specific requirement/source paths> and applicable repository instructions.
Any stale document status must be distinguished from current code/test evidence.

In scope:
- <specific production change and its allowed paths>
- <necessary integration and compatibility behavior>
- <required unit tests and allowed test paths>

Out of scope:
- <adjacent tasks/features that must remain unstarted>
- Unrelated cleanup, speculative abstractions and performance work.
- Documentation writing, formatting or validation, tracker edits, HTML/Safari
  work, link audits and document hash/repeat-generation cycles.

SELECTED TEAM
Main: coordinate, define the interface, integrate, resolve findings and decide
completion. Do not start later tasks.

Implementer A: <turbo-mac-implementer OR grok-implementer>
Coding responsibility: <concrete behavior>.
Exclusive production paths: <exact paths>.

Implementer B: <none OR the other implementer>
Coding responsibility: <separate concrete behavior, or omit if none>.
Exclusive production paths: <non-overlapping paths, or omit if none>.

Unit tests: turbo-luna-test-engineer
Exclusive test/fixture paths: <exact paths>.
Write meaningful tests from the requirements and run focused checks when Main
grants the serial execution slot. Report failures to Main. Do not edit production
code or weaken tests to accommodate defects.

Reviewer: <turbo-mac-review OR independent-reviewer>
READ-ONLY. Read the implementation, relevant tests and existing test results.
Return concrete findings/advice or explicitly no findings to Main, with file,
location, triggering case and impact. Check scope and preserved behavior.
State what could not be verified. Do not edit production code, tests, fixtures
or documents. Do not run builds/tests or delegate work.

Use only this selected team and its configured models. Grok and the expensive
independent-reviewer are authorized only if explicitly selected above.
Agents share the checkout. Preserve others' edits and do not overwrite or revert
unrelated changes. Main resolves ownership conflicts before further edits.

ACCEPTANCE
- <required normal behavior and how it is checked>
- <boundary/failure behavior and how it is checked>
- <compatibility requirement and how it is checked>
- <resource/cancellation/security/numerical condition, only if relevant>

EXECUTION LIMITS
Allowed operations: <for example: code edits and tiny synthetic unit tests only>.
Prohibited operations: <task-specific restrictions, including real weights/model
execution/live settings if not explicitly authorized>.
Follow applicable repository preflights. Use Scripts/test.sh for package tests.
Main coordinates one build/test process at a time. Select focused tests positively
and exclude <specific original-checkpoint/model tests outside this task>.
Do not commit, push, deploy, create worktrees or change agent configuration.
No timers, continuous monitoring or automatic next-task dispatch.

WORKFLOW AND COST
1. Inspect the relevant current code/diff once. State the small implementation
   contract and file ownership in chat, then dispatch the selected coding roles.
2. Let the test engineer add independent tests against the agreed behavior.
   Only run once the relevant edits are ready and the execution slot is free.
3. Send the final implementation and existing test evidence to the selected
   reviewer. Main routes production fixes to implementers and test fixes to Luna.
4. Rerun only affected tests and recheck affected findings after relevant changes.
   Do not repeatedly audit unchanged work or request duplicate reviews.
5. If an agent stalls, inspect its last action/error before retrying. Avoid repeated
   blind retries. Report a concrete blocker if the selected team cannot proceed.

Keep only necessary technical evidence in <repository-owned evidence directory>:
changed paths, exact commands, actual test counts/failures/skips/exit codes and
enough file identity to show the tested code matches the final candidate.
Use existing runner output. Do not build a new reporting system or polished report.

DONE AND STOP
Complete only when the requested behavior is implemented, required tests pass,
and Main has resolved blocking review findings against the final code.
Document formatting/publication is not a completion gate for this technical run.
Do not claim deferred documents or unrun checks passed.

Return a short chat summary: what changed, test results, reviewer advice and
Main's decision, plus any remaining technical blocker. Then STOP.
Do not begin <next task ID/name> without Hebert's explicit request.
```

If switching agents during an active task, send a short correction to Main naming the old role, new owner, exact paths and handoff point. Preserve completed work and confirm the previous writer has stopped before the replacement edits those files. Do not restart completed tests or reviews solely because the owner changed.
