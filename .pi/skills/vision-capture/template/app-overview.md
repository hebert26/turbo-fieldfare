# Overview — `<App Name>`

Authoring template. Copy to `reference/<app-folder>/overview.md`. Describe only
verified purpose and capability areas. No feature may appear here unless a README,
source file, or built artifact proves it. Delete anything you could not verify.

This file is context. It is never execution authority and never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `<exact.bundle.id>` | `<file>` |
| Display name | `<CFBundleDisplayName>` | `<Info.plist>` |
| Version | `<CFBundleShortVersionString>` | `<Info.plist>` |
| Minimum OS | `<MinimumOSVersion>` | `<Info.plist>` |
| Device families | `<iPhone / iPad>` | `<project config>` |

## Purpose

`<One or two sentences on what the app is for, taken from its README or product doc.
Cite the file.>`

## Capability areas

| Area | What it covers | Verified from |
|---|---|---|
| `<area>` | `<short description>` | `<file>` |

List areas, not screens-by-label. An area is a capability the app declares in source
or documentation.

## System permission prompts to expect

`<List the usage-description keys found in the built Info.plist and the source call
sites that request them. If there are none, say "None — no usage-description keys in
the built Info.plist." Never guess which prompts appear.>`

## State and reset behaviour

`<Record only verified facts: persisted flags, first-run gates, mock vs real data,
network use. Cite the source file for each.>`

## Source-declared accessibility landmarks

These are identifiers declared in source. They exist to help you **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared in |
|---|---|
| `<identifier>` | `<file:line>` |

Mark any runtime-composed identifier (built from a title, index, or other runtime
value) as unstable — it is not a reliable landmark.

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
