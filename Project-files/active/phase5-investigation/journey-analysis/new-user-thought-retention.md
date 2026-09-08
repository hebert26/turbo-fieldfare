# New-user thought retention: narrow source follow-up

Current app code retains raw generated thought tokens when a later user message starts in the same conversation. It does not rebuild a thought-stripped history at that boundary. This is separate from its correct retention of context through successive results in one tool loop.

Scope: read-only static examination of `/Users/dev-machine/dev/turbo-fieldfare-personal` on 2026-09-07, plus the already frozen screenshot-diagnostic trace. No app control, model run, source edit or test. The parent owns verification of Google's expected turn-boundary semantics. This note establishes current local behavior.

## Exact path

1. `AppModel.run`, `Sources/TurboFieldfareApp/Core/State/AppModel.swift:1943`, calls `conversation.beginTurn`. `AppConversation.beginTurn`, `Sources/TurboFieldfareApp/Core/Inference/AppConversation.swift:97–104`, returns the existing conversation epoch, without replacing it. The actual runtime reset in `AppModel.openConversationIfNeeded:1931–1935` occurs only when `serviceEpoch != conversation.epoch`. `newChat:1876,1888` changes that state explicitly. A normal later user turn does not.
2. `RealInferenceClient.conversationForTurn`, `Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift:537–547`, returns the existing `MultimodalConversation`. Its user-request branch at `608–623` calls `sendToolUser`; a tool-result branch at `626–644` calls `sendToolResults`.
3. `MultimodalConversation.sendToolUser`, `Sources/TurboFieldfare/Runtime/Generation/MultimodalConversation.swift:460–468`, checks the tools and pending-result state, appends a user message to its semantic message list, then calls `encodeTurn`. It does not remove thought spans, reset the runner or mark KV for rebuilding. Its own comment at `419–422` describes later user turns as KV continuations.
4. `encodeTurn:994–1000` uses `tokenizer.encodeTextContinuation(userContent:)` when the token history is nonempty. `Sources/TurboFieldfare/Tokenization/Tokenizer.swift:430–435` creates only the end-of-turn/new-user/new-model suffix. It cannot strip or rewrite prior token history. The image continuation path at `MultimodalConversation.swift:1015–1019` likewise appends a new turn against the existing conversation.
5. `completeEncodedTurn:668` builds `promptIDs = kvTokenIDs + boundary + turn.effectiveTokenIDs`. At `674`, all existing token IDs are treated as cached unless a separate recovery condition already marked them for rebuilding. At `790`, generation resumes the cached KV when this count is nonzero. At `933`, it saves `result.kvBackedTokenIDs` again.
6. `RawCompletion`, `Sources/TurboFieldfare/Runtime/Generation/RawCompletion.swift:163`, initializes history from the retained prompt prefix. At `312–315`, every committed generated token enters history and the next forward pass regardless of the channel displayed by the parser. At `327`, that history becomes `kvBackedTokenIDs`.

## Display is a separate projection

`MultimodalConversation.swift:802–837` feeds raw tokens to `StructuredAssistantDecoder`, then appends only `.content` events to visible text. `StructuredAssistantDecoder.swift:201–217` suppresses thought-channel content from those visible events. `MultimodalConversation.assistantMessage:625–635` stores the visible answer or parsed calls in the semantic message list. That filtered semantic list is not used to reconstruct the full prefix for the next user message in `sendToolUser`'s existing-state branch.

The stop-string trimming at `MultimodalConversation.swift:935–958` removes a matched trailing stop-string span. It is not general thought removal and does not run at every new-user boundary. Explicit `reset:301–326` removes the entire token history and tool state. Neither mechanism supplies thought-stripped continuation for an ordinary later user turn.

## Existing trace corroboration

`image_diagnostic.snapshot.jsonl` is the frozen extended diagnostic, SHA-256 `2d98ef9752998ced950e7d9fe4b4f9faad71394d6f6a16d1055747cb23357501`.

- Line 3: first user request ends naturally with `Ready.`, `structured_progress.thinking_tokens = 60`, and `diagnostics.conversation_tokens = 1639`.
- Line 4: a distinct new user request begins (`turn_index = 1`, `input.kind = user`).
- Line 5: that request's output reports `cached_prompt_tokens = 1639` and `computed_prefill_tokens = 78`. The exact full previous context count was retained across the user boundary.
- Lines 8–9 provide another new-user boundary, retaining the previous 2,135 context tokens as `cached_prompt_tokens = 2135`.

This demonstrates full-count retention in the historical diagnostic. The trace does not dump the actual KV token IDs, so the raw-thought identity conclusion is supplied by the source path above. Different historical binaries must not be assumed identical to current dirty source solely from the trace.

## File hashes read during this check

| Current dirty source | SHA-256 |
|---|---|
| `Runtime/Generation/MultimodalConversation.swift` | `ddd8eb4754236db4f2aa7ecba80876dda91f344b2d030fc11460b3e1d49b08c0` |
| `Runtime/Generation/RawCompletion.swift` | `f48637163ba6af7f6c81792df0390ea41d53e933ee236ab189af72ca89ef45e3` |
| `Tokenization/Tokenizer.swift` | `19e33f0739d65e846cd5cc7a89884ffd24e7f9ccedecd4ba704be77ddc725547` |
| App `Core/State/AppModel.swift` | `c21ec6e97a907dc2f7dabfa57698f449d438b42508913e8bc15053ba25ad2c2b` |

Commands were `rg -n` for the named functions, `nl -ba ... | sed -n ...` for the cited ranges, and `shasum -a 256` for the four files, all read-only. Some exploratory guessed paths returned missing-file status before `rg --files` identified the actual files. All cited reads and final hash command exited 0.

Recommendation: resolve the expected new-user thought-history contract before selecting an optimization. Preserve raw thought context within the same tool loop. Any eventual turn-boundary implementation must keep token IDs, KV positions, retained image-span offsets and the semantic history consistent. No implementation or measured quality/performance benefit is claimed here.
