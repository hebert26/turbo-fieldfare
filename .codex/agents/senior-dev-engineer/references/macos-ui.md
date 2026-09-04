# macOS UI Reference

Use this for SwiftUI/AppKit UI work, visual polish, app shell, settings, windows, commands, toolbars, accessibility,
and macOS-native behaviour.

## Owned Areas

- `VisionCapture/Sources/App/`
- `VisionCapture/Sources/UI/`
- `VisionCapture/Sources/Features/`

You own:

- SwiftUI view composition;
- macOS toolbar, sidebar, inspector, sheet, menu, and window patterns;
- narrow AppKit bridges for missing desktop capabilities;
- Liquid Glass and modern macOS styling where the codebase uses it;
- animations and motion that help the task;
- accessibility labels, focus order, keyboard reachability, and contrast;
- responsive behaviour for resizable windows.

## Working Rules

- Keep behaviour intact unless the user asks for behaviour changes.
- Preserve menu and shortcut access when touching toolbar actions.
- Use existing theme and component conventions.
- Avoid app-specific automation examples in UI copy.
- Test visible states: ideal, empty, loading, error, and success where relevant.
- Model Mac scenes explicitly: main window, settings, utility windows, inspectors, menu bar extras.
- Prefer native macOS structure before custom chrome.
- Use AppKit only for the smallest missing desktop capability.
- Keep SwiftUI as the source of truth for state.

## Accessibility

- Add stable accessibility identifiers on new or modified SwiftUI views.
- Use the project format, for example `screen.<feature>.<screen>.root` and `ui.<feature>.<name>Button`.
- Keep identifiers stable: no runtime values, array indices, or localized strings.
- Preserve focus, labels, keyboard reachability, and contrast.

## Verification

Run `swift build` after UI code changes. Use previews or manual app verification where practical. If a change affects
accessibility or resizing, say how it was checked.
