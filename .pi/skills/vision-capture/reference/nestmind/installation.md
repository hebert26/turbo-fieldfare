# Installation — NestMind (debug)

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.hebertgo.nestmind.debug` | built `Info.plist` → `CFBundleIdentifier` |
| Source project | `/Users/dev-machine/Dev/NestMind` | directory listing |
| Xcode project | `/Users/dev-machine/Dev/NestMind/NestMind.xcodeproj` | directory listing |

Only the `.debug` bundle is a test target. `com.hebertgo.nestmind` (release) is a
different app and is never the target. The `Debug` configuration is what produces the
`.debug` bundle ID — do not build `Release` for a test run.

No simulator UDID is recorded here. A UDID is not app truth. Use the caller's current
target device.

## Build and install

**No prebuilt `.app` path is recorded.** A stale artifact passes every check a
document can make — the directory exists, its `CFBundleIdentifier` matches — so
neither proves freshness. Only a fresh build proves its own. Build it.

Verified build command:

```bash
cd /Users/dev-machine/Dev/NestMind
./scripts/build-app-only.sh
```

Its defaults, read from the script: scheme `NestMind`, configuration `Debug`,
destination `generic/platform=iOS Simulator`, derived data `./build/DerivedData`,
project `./NestMind.xcodeproj`. It builds only — no install, no tests. Add `--clean`
for a clean build.

Your build writes the artifact to:

`/Users/dev-machine/Dev/NestMind/build/DerivedData/Build/Products/Debug-iphonesimulator/NestMind.app`

That path is where *your* build just wrote. It is not a prebuilt artifact to reuse —
if you did not just build, do not install from it.

Before `install app`, verify the artifact:

```bash
/usr/libexec/PlistBuddy -c "Print :CFBundleIdentifier" \
  "/Users/dev-machine/Dev/NestMind/build/DerivedData/Build/Products/Debug-iphonesimulator/NestMind.app/Info.plist"
```

It must print exactly `com.hebertgo.nestmind.debug`.

If the build fails, stop and report the failure. Do not install an older artifact, do
not invent another build command, do not glob DerivedData, and do not fall back to the
release bundle.

## Install sequence

A live test, post-fix validation, or acceptance run must start from a fresh install —
see the fresh-cycle rule in [`../../SKILL.md`](../../SKILL.md). If `launch app`
succeeds because the app is already there, that precondition is not met: stop and ask
the operator to delete the app from that Simulator. VisionCapture has no public
`uninstall app` request, and raw `simctl` is not a substitute.

1. The first VisionCapture operation for a requested app is `launch app`, with the
   caller's exact `bundle_id` and `udid`. Do not screenshot, `describe screen`, or
   `inspect cache` first.
2. `APP_NOT_INSTALLED` proves the target is not installed on that Simulator. Stop
   navigation — no app action can run until it is installed.
3. Build the app with the command above, then verify the artifact it produced. Its
   `CFBundleIdentifier` must print exactly `com.hebertgo.nestmind.debug`. If the build
   fails or the identifier does not match, stop and report it.
4. Only then send `install app` with that absolute `app_path` and the same `udid`.
   Require the returned `bundle_id` to equal `com.hebertgo.nestmind.debug`.
5. After install, send `launch app` again.
