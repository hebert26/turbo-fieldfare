# Overview — MuckIdentity

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.identity` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckIdentityApp.swift`, `Sources/MuckIdentityModels.swift`, `Sources/MuckIdentityRoot.swift` (234 lines) | file listing |

## Purpose

A mock local sign-in flow with a three-step wizard (credentials, code verification, completion) and a recovery-address sheet, explicitly not connected to any identity service.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Credentials step | an account-name field and a secure passcode field | MuckIdentityRoot.swift:72 |
| Code verification | a four-digit fixture code entry checked against a fixed value | MuckIdentityRoot.swift:92, MuckIdentityModels.swift:53 |
| Recovery | a sheet to enter a recovery address and prepare a local note | MuckIdentityRoot.swift:36 |
| Completion | a success state with a start-over action | MuckIdentityRoot.swift:111 |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `secure-fields,code,validation`; navigation
`signin-wizard-recovery-modal`; motion `error-shake`; accessibility profile
`B`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckIdentityState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

14 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.identity.root` | MuckIdentityRoot.swift:35 |
| `state.identity.complete` | MuckIdentityRoot.swift:116 |
| `state.identity.message` | MuckIdentityRoot.swift:27 |
| `ui.identity.accountNameField` | MuckIdentityRoot.swift:78 |
| `ui.identity.backButton` | MuckIdentityRoot.swift:107 |
| `ui.identity.closeRecoveryButton` | MuckIdentityRoot.swift:53 |
| `ui.identity.continueButton` | MuckIdentityRoot.swift:85 |
| `ui.identity.passcodeField` | MuckIdentityRoot.swift:81 |
| `ui.identity.prepareRecoveryButton` | MuckIdentityRoot.swift:47 |
| `ui.identity.recoveryAddressField` | MuckIdentityRoot.swift:43 |
| `ui.identity.recoveryButton` | MuckIdentityRoot.swift:88 |
| `ui.identity.startOverButton` | MuckIdentityRoot.swift:120 |
| `ui.identity.verificationCodeField` | MuckIdentityRoot.swift:100 |
| `ui.identity.verifyButton` | MuckIdentityRoot.swift:104 |

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
