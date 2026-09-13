---
name: using-gpudebug
description: ALWAYS use when a user has a .gputrace file and needs to inspect, navigate, or extract resources from it using the gpudebug CLI. Covers draw calls, render encoders, pipeline state, resource bindings, textures/buffers, and shader source. Also trigger when another skill needs to read data from a .gputrace (fetch textures, inspect state, list encoders). Do NOT trigger for capturing a new trace — use using-gpucapture.
---

# Using gpudebug — Metal GPU Trace Debugger

## Overview

`gpudebug` is a CLI debugger for Metal GPU traces. It opens a `.gputrace` file and provides commands for navigating the captured API call hierarchy, inspecting resources and pipeline state, fetching textures and buffers to disk, and browsing shader source code.

## Step 1: Check availability

```sh
which gpudebug
```

`gpudebug` ships with macOS 27 and later. If not found, tell the user it is not available and suggest opening the `.gputrace` in Xcode's Metal Debugger instead: [Analyzing your Metal workload](https://developer.apple.com/documentation/Xcode/Analyzing-your-Metal-workload)

If the user explicitly asks to inspect the trace in Xcode, honor their request — point them to the Apple docs URL above and answer from your own knowledge. Do not override their choice with `gpudebug`.

## Step 2: Read the man page

The man page is the authoritative reference for all commands, options, the trace hierarchy, and node naming conventions. Read it before using gpudebug:

```sh
man gpudebug | col -b
```

The man page covers: CLI flags, the trace tree structure, node naming (prefixes and indexing), all REPL commands (`list`, `go`, `info`, `fetch`, `find`, `next`/`prev`, `status`), scripting examples, and tips.

## Step 3: Agent usage

Always drive gpudebug **non-interactively** — never start an interactive REPL. Use `-q` to suppress the startup banner. The man page documents the full scripting model, command syntax, and examples.

A background replayer connects to the GPU on startup. Browse commands (`list`, `go`, `info`, `find`) work immediately, but `fetch` requires the replayer to be ready. Structure scripts so browse operations come first — by the time you need to `fetch`, the replayer has usually initialized. If the replayer failed (e.g., trace captured on a different GPU architecture), static browsing still works but `fetch` is unavailable.
