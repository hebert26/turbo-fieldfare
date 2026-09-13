---
name: using-gpucapture
description: ALWAYS use when a user wants to capture a GPU trace (.gputrace) of a Metal application. Trigger when the user wants to capture a GPU trace, get a gputrace, record a Metal frame, capture a frame, capture GPU workload, use gpucapture, capture from the command line, do a Metal frame capture, ask how to capture a Metal trace, capture a running process, or otherwise create a .gputrace file from a Metal app. Do NOT trigger when the user already has a GPU trace and wants to analyze or diagnose it — use using-gpudebug for that.
---

# Using gpucapture — Metal GPU Trace Capture

## Overview

`gpucapture` is a CLI tool for capturing Metal GPU traces from running processes or launched executables. It requires no source code changes — attach to a process by PID, choose what to capture (frames, command buffers, custom scopes), and produce a `.gputrace` file for analysis.

## Step 1: Check availability

```sh
which gpucapture
```

`gpucapture` ships with macOS 27 and later. If not found, the user is on an older macOS version — suggest alternatives based on their situation:

- **Can modify source code →** Programmatic capture via `MTLCaptureManager`: [Capturing a Metal workload programmatically](https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-programmatically)
- **Has Xcode available →** Xcode's built-in GPU Frame Capture: [Capturing a Metal workload in Xcode](https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-in-xcode)

If the user explicitly asks for programmatic capture or Xcode capture, honor their request — point them to the relevant Apple docs URL above and answer from your own knowledge. Do not override their choice with `gpucapture`.

If `gpucapture` is available and the user has not specified a method, proceed with the steps below.

## Step 2: Read the man page

The man page is the authoritative reference for all subcommands, flags, boundary types, and environment variables. Read it before using gpucapture:

```sh
man gpucapture | col -b
```

## Step 3: Key requirements

The man page documents all subcommands (`list`, `boundary`, `start`, `stop`), flags, and boundary types. Two things to keep in mind that are easy to miss:

- **The target process must be launched with `MTL_CAPTURE_ENABLED=1`.** There is no way to enable capture on an already-running process that was launched without it. If the user says the app is "already running," they likely need to relaunch it.
- **For apps that exit quickly**, `MTLCAPTURE_WAIT_FOR_SIGNAL=1` pauses the process at `MTLDevice` creation until `gpucapture start` attaches. Both env vars should be set at launch.

After capturing, verify the `.gputrace` file was created and report the file path and size to the user.
