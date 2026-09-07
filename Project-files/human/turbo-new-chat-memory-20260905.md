# Completed chat memory check

The New Chat control cleared the completed Journey 5 transcript and error. A short memory sample showed app footprint moving from 223M to 218M. This does not establish a leak, complete allocation release, or long-run memory stability.

## Observed setup

On 2026-09-05 at 08:41–08:42 Europe/London, the installed TurboFieldfare app PID 68969 was `Installed · Not loaded`. Decode service PID 69064 remained alive. No generation or target-app navigation was started. The existing Journey 5 trace and receipt preserve the completed test before its chat was cleared.

The computer-use accessibility read showed the old Journey 5 prompt, the stale-revalidation error, and the unloaded model state. Root clicked Clear Input and then the newly visible New chat button. The next read showed an empty composer, no conversation transcript or error, and the same unloaded state.

## Measurements

Command before clearing: `top -l 1 -R -F -pid 68969 -pid 69064 -stats pid,command,mem,pageins`, exit 0. At 08:41:24, app footprint was 223M and service footprint 347M.

Command after clearing: `top -l 2 -s 5 -R -F -pid 68969 -pid 69064 -stats pid,command,mem,pageins`, exit 0. At both 08:42:20 and 08:42:25, app footprint was 218M and service footprint 347M.

These are the `top` MEM values, not resident-set size. The model had already been unloaded before the first sample. Menu operations and composer clearing occurred between samples. There was no simultaneous allocation capture, so the 5M difference cannot be attributed entirely to transcript disposal. The unchanged service measurement does not by itself identify retained model allocations.

## Source finding and deployed correction

Astra found that `InstructionTranscriptDocumentController.resetTranscript` cleared the transcript and progressive rendering state but retained `ProgressiveState.cache`. That dictionary owns full text keys and rendered attributed strings. Its entry count is capped at 512, but it had no whole-chat release. The measured footprint does not establish its retained byte count. Journey 5 had empty assistant text, so this cache is not established as the cause of its app footprint.

The correction adds `progressive.cache.removeAll(keepingCapacity: false)` only inside the whole-transcript reset. Normal turn and streaming resets continue to reuse cached rendering. The model unload path already releases conversation, model, vision runtime, runner, scratch and tokenizer references. Its reusable Metal context remains service infrastructure.

Astra implemented the source change. Root reviewed the reset body, quit the idle app normally, verified no model process remained, and ran `./script/build_and_run.sh --verify`, exit 0. Build log: `/tmp/turbo-chat-cache-reset-build-20260905.log`. Release app build took 13.26 seconds, service build 0.45 seconds. The existing script verified signatures and equality of staged and installed binaries, then launched `/Applications/TurboFieldfare.app`.

Installed app SHA-256: `3bd663e1480a98268c66265ae8127b4e4b1dde82c054bf22aa7c2f58789a049a`. Installed service SHA-256: `d4f709d5ca670ed0dd4411554009bb26f89032b4c5f59fc77bfcc4ae0b8b3adb`.

Root verified the opened app through computer use: empty chat, Installed · Not loaded, 64K context and 32 slots. The verified NestMind bundle identifier and simulator UDID were restored in the UI. No model generation or full exploration was run on this build. The correction's ownership behavior is supported by source inspection and compilation; its live retained-byte reduction and effect on long-run memory remain unmeasured. No token-speed improvement is claimed.
