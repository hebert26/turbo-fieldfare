# Capture And OCR Reference

Use this for screenshots, Simulator window discovery, PNG handling, coordinate metadata, and OCR.

## Owned Areas

- `VisionCapture/Sources/CaptureCore/`
- `VisionCapture/Sources/OCRCore/`
- Core call sites that consume capture or OCR outputs.

You own:

- CGWindowList capture and Simulator window discovery;
- `simctl` screenshot paths;
- PNG encoding and decoding boundaries;
- Vision framework OCR behaviour;
- coordinate metadata tied to captured images;
- capture error classification;
- keeping `CaptureCore` and `OCRCore` dependency-light.

## Boundaries

- `CaptureCore` and `OCRCore` should stay free of external package dependencies.
- Do not own UI design, MCP routing, WDA actions, or persistence schemas unless capture output contracts change.
- Do not bake in a specific Simulator, app, or screen.

## Working Rules

- Avoid blocking the main actor with image work.
- Preserve Sendable value boundaries across actors.
- Keep memory use visible when handling large images.
- Keep capture/OCR errors clear enough for diagnosis.
- Preserve metadata needed for coordinate translation.

## Verification

Run `swift build` from `VisionCapture/`. Run focused CaptureCore or OCR tests. For image changes, report whether a
real capture or fixture-based check was used.

## Handoff Notes

Report:

- capture or OCR path changed;
- dependency impact;
- main-actor and memory impact;
- tests or manual capture checks;
- remaining environment risks.
