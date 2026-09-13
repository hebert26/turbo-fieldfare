---
name: using-metal-validation
description: ALWAYS use when a user wants to enable, configure, or run Metal validation — API validation (MTL_DEBUG_LAYER) for CPU-side API misuse, shader validation (MTL_SHADER_VALIDATION) for GPU-side memory errors, or the load/store-action visual indicators. Trigger when working with Metal validation (MTL_DEBUG_LAYER, MTL_SHADER_VALIDATION, the validation layer, API validation, MTL_DEBUG_LAYER_VALIDATE_LOAD_ACTIONS, MTL_DEBUG_LAYER_VALIDATE_STORE_ACTIONS, per-pipeline validation, MTL4PipelineOptions.shaderValidation, validation perf hit), or when encountering symptoms like shader validation errors not appearing, fuchsia color render artifacts, the red/white checkerboard, or zerofill output. Do NOT trigger for general rendering bug diagnosis where validation isn't the relevant tool.
---

# Using Metal Validation

## Overview

Metal provides two validation modes, both configured via environment variables read at `MTLDevice` creation:

- **API validation** (`MTL_DEBUG_LAYER`) — CPU-side. Cheap; enable broadly during development.
- **Shader validation** (`MTL_SHADER_VALIDATION`) — GPU-side. Potentially significant perf cost; enable selectively or per-pipeline.

Env vars must be set **before** any `MTLDevice` is created. Changes after device creation have no effect.

## Step 1: Read the man page

The man page is the authoritative reference for all env vars, their valid mode values, and pipeline scoping syntax. Read it before configuring anything beyond the quick-start:

```sh
man MetalValidation | col -b
```

## Step 2: Quick-start

```sh
open -a MyGame.app \
    --env MTL_DEBUG_LAYER=1 \
    --env MTL_SHADER_VALIDATION=1 \
    --env MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1
```

## Step 3: Things easy to miss

The man page documents every variable, but a few workflow points are easy to miss:

- **Launch `.app` bundles via `open --env`, instead of invoking the binary inside the bundle directly** (e.g., `MyGame.app/Contents/MacOS/MyGame`). Use `open -a MyGame.app --env KEY=VALUE` (one `--env` per variable) for every run that needs validation.
- **Shader validation errors go to the system log by default, NOT stderr.** Without `MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1`, errors are invisible from a terminal-launched build. Always pair the two.
- **Per-pipeline scoping for ship-grade builds.** If global `MTL_SHADER_VALIDATION=1` is too expensive, set `MTL4PipelineOptions.shaderValidation = MTLShaderValidationEnabled` on the descriptor for shaders under active iteration. Env-var allow-list / deny-list (`MTL_SHADER_VALIDATION_ENABLE_PIPELINES` / `_DISABLE_PIPELINES`) is the equivalent for pre-Metal-4 codebases.
