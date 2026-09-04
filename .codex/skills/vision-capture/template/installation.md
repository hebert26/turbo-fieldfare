# Installation — `<App Name>`

Authoring template. Copy to `reference/<app-folder>/installation.md`, replace every
`<placeholder>` with a value verified on this machine, and delete any row you could
not verify. Never invent a path, a bundle ID, or a build step.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `<exact.bundle.id>` | `<file or command>` |
| Source project | `<absolute path>` | `<file or command>` |
| Xcode project | `<absolute path to .xcodeproj>` | `<file or command>` |

Do not record a simulator UDID here. A UDID is not app truth. Always use the caller's
current target device.

## Build and install

**Never record a prebuilt `.app` path.** A stale artifact passes every check a
document can make — the directory exists, its `CFBundleIdentifier` matches — so
neither proves freshness. Record the *build command* instead. A fresh build is the
only artifact whose freshness is provable.

Find the project's real build command and verify it before recording it. Read the
script; if it supports `--dry-run`, run that and paste the exact command it prints.
Never invent a command and never record one you have not confirmed.

Fill in this shape:

> **No prebuilt `.app` path is recorded.** … Only a fresh build proves its own. Build
> it.
>
> Verified build command:
>
> ```bash
> cd <project root>
> <the verified build command>
> ```
>
> Its defaults, read from the script: scheme `<scheme>`, configuration `<config>`,
> destination `<destination>`, derived data `<path>`.
>
> Your build writes the artifact to:
>
> `<artifact path>`
>
> That path is where *your* build just wrote. It is not a prebuilt artifact to reuse —
> if you did not just build, do not install from it.
>
> Before `install app`, verify the artifact's `CFBundleIdentifier` prints exactly
> `<exact.bundle.id>`.
>
> If the build fails, stop and report the failure. Do not install an older artifact,
> do not invent another build command, and do not glob DerivedData.

If you cannot verify a build command for this app, say so plainly and tell the caller
to ask the customer for the exact built `.app` path after `APP_NOT_INSTALLED`.

## Install sequence

1. The first VisionCapture operation for a requested app is `launch app`, with the
   caller's exact `bundle_id` and `udid`. Do not screenshot, `describe screen`, or
   `inspect cache` first.
2. `APP_NOT_INSTALLED` proves the target is not installed on that Simulator. Stop
   navigation — no app action can run until it is installed.
3. Ask the customer for the exact built `.app` path. Verify what they give you before
   using it — both checks must pass:

   ```bash
   ls -d "<app-path>"
   /usr/libexec/PlistBuddy -c "Print :CFBundleIdentifier" "<app-path>/Info.plist"
   ```

   The directory must exist, and `CFBundleIdentifier` must equal the requested bundle
   ID exactly. If either check fails, go back to the customer.
4. Only then send `install app` with that absolute `app_path` and the same `udid`.
   Require the returned `bundle_id` to equal the requested bundle ID.
5. After install, send `launch app` again.
