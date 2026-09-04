# Cleanup

**Cleanup happens once, at the end of all phases. It is the last task of the last phase.**

Agents build, test, boot simulators, clone devices and create worktrees. None of that gets removed on its own. A long work item can fill the disk in a few days, and then every later phase fails for a reason that has nothing to do with the code.

Measured on this machine, 2026-08-15:

| What | Size |
|---|---|
| `~/Library/Developer/CoreSimulator/Devices` | **33 GB** |
| `~/Library/Logs/CoreSimulator` | **2.7 GB** |
| `VisionCapture/.build` | **2.4 GB** |
| `~/Library/Developer/Xcode/DerivedData` | **1.5 GB** |
| stale agent worktrees under `~/.codex/worktrees/` | 4 of them |

Free space at the time: **35 GB**. One more work item of that size and the disk is gone.

## The task

Last line of the last phase:

```
- [ ] <n>.<last> Clean up: build output, logs, devices and worktrees this work item created
```

It is a real task with a real checkbox. It counts for D1 on that phase, so **the work item does not close until the disk is given back.**

Earlier phases carry no cleanup task. They still record a disk baseline the moment they start - see recipe 0 in `update-recipes.md` - because this final cleanup needs it to tell what the work item created from what was already there.

## Before you delete anything

Measure. Put the numbers in the implementation document's phase evidence, so the next person can see what the phase actually cost.

```sh
df -h / | tail -1
du -sh VisionCapture/.build 2>/dev/null
du -sh ~/Library/Developer/Xcode/DerivedData 2>/dev/null
du -sh ~/Library/Developer/CoreSimulator/Devices 2>/dev/null
du -sh ~/Library/Logs/CoreSimulator 2>/dev/null
```

## Never delete

Read this list before you delete anything. Getting it wrong costs someone a day.

- **Evidence folders.** `evidence/` is the proof for D5. It is not clutter.
- **Any simulator device this phase did not create.** On this machine that includes `DO-NOT-TOUCH`, `NestMind Customer 1`, `NestMind Customer 2`, and anything else already booted when the phase started.
- **Any worktree with uncommitted changes.** Another agent may still be working in it.
- **The default device set.** Only remove device sets this phase generated.
- **Anything under `project-files/`.** That is the work, not the waste.

If you are not certain you created it, leave it and say so in the evidence line.

## What to clean

### 1. Build output

```sh
cd VisionCapture && swift package clean
```

Or remove it outright when the phase is finished with it:

```sh
rm -rf VisionCapture/.build
```

Costs a full rebuild next time. Worth it at the end of a phase, not between two tasks in the same phase.

### 2. DerivedData

Only the folders for this project:

```sh
ls -d ~/Library/Developer/Xcode/DerivedData/VisionCapture-* 2>/dev/null
# check the list, then:
rm -rf ~/Library/Developer/Xcode/DerivedData/VisionCapture-*
```

Never `rm -rf ~/Library/Developer/Xcode/DerivedData` whole - other projects live there.

### 3. Simulator devices this phase created

List first. Compare against the baseline the phase recorded when it started.

```sh
xcrun simctl list devices booted
xcrun simctl list devices | grep -i '<the name pattern this phase used>'
```

Then, **only for devices this phase created**:

```sh
xcrun simctl shutdown <UDID>
xcrun simctl delete <UDID>
```

If the phase generated its own device set:

```sh
xcrun simctl --set <path-to-generated-set> delete all
rm -rf <path-to-generated-set>
```

This is the 33 GB. It is also the easiest thing to get wrong. One device is roughly 2 GB, so leaving five behind costs 10 GB.

### 4. Simulator logs

```sh
du -sh ~/Library/Logs/CoreSimulator
rm -rf ~/Library/Logs/CoreSimulator/*
```

Safe. These are logs, not state. 2.7 GB sits there right now.

### 5. Stray processes

Helper processes and test runners survive a crashed run and hold memory.

```sh
pgrep -fl '<helper or runner name this phase started>'
# check the list, then kill only what this phase started
```

Never kill `Simulator`, `com.apple.CoreSimulator.CoreSimulatorService`, or anything you did not start.

### 6. Agent worktrees

```sh
git worktree list
git worktree prune
```

`prune` only removes entries whose folder is already gone - it is safe. To remove a real one, check it first:

```sh
git -C <worktree path> status --porcelain   # must be empty
git worktree remove <worktree path>
```

If `status` prints anything, leave it. Someone has unsaved work.

### 7. Scratch files

Delete the temp files this phase made outside the repo. Leave the work-item folder alone.

## Record it

In the implementation document, under the phase evidence table:

```
| 2026-08-15 | cleanup | freed 6.2 GB - .build, DerivedData, 3 generated devices, sim logs | evidence/2026-08-15_cleanup/cleanup.txt |
```

If something was deliberately left behind, say what and why:

```
Left: worktree ~/.codex/worktrees/378b - has uncommitted changes, not ours.
```

## When the disk is already tight

Cleanup being one task at the end does not mean nobody may clean before then.

If `df -h /` shows under 20 GB free, stop and clean before starting the next phase. Do not begin work you cannot finish. Use the same "never delete" list and the same measure-first commands.

An early clean like that is housekeeping, not the cleanup task. It ticks nothing, closes nothing, and does not remove the final cleanup task from the last phase. Record it as a line in that phase's evidence table so the final cleanup knows what is already gone.

Tell the owner the number. Running out of disk mid-phase corrupts builds and wastes more time than the cleanup ever would.
