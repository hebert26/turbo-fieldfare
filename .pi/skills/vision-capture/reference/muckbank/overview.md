# Overview — MuckBank

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.bank` | built `Info.plist`; `project.yml` |
| Display name | `MuckBank` | built `Info.plist` → `CFBundleDisplayName` |
| Version | `1.0` | built `Info.plist` |
| Minimum OS | `17.0` | built `Info.plist` → `MinimumOSVersion` |
| Device families | iPhone and iPad | `project.yml` → `TARGETED_DEVICE_FAMILY: "1,2"` |

## Purpose

A mock iOS banking app, one of the MuckApps fixtures built for VisionOS automation
and UI interaction testing (`/Users/dev-machine/Dev/MuckApps/README.md`). It is a
fixture, not a customer product.

All of it lives in one file: `Sources/MuckBankApp.swift` (1,399 lines). It uses only
in-memory mock data and makes no network calls (`MuckApps/README.md`).

## Capability areas

Root is a three-tab `TabView` (`MuckBankApp.swift:353`), each tab a `NavigationStack`.

| Area | What it covers | Verified from |
|---|---|---|
| Accounts | Account list, animated balance, balance visibility toggle, account detail, transaction list and detail, dispute | `AccountsOverviewView:412`, `AccountRow:499`, `AccountDetailView:539`, `TransactionRow:637`, `TransactionDetailView:676` |
| Transfer | Multi-step transfer wizard: source/destination pickers, amount field and slider, quick amounts, memo, validation, review, confirm, success | `TransferFlowView:746` |
| Cards | Card tiles, card detail, freeze / online-payments / ATM / contactless toggles, spending-limit slider, report-lost and order-replacement dialogs | `CardsView:1130`, `CardFaceView:1155`, `CardDetailView:1220` |
| Activity | Activity list with a reset action behind a confirmation | `ActivityView:1351` |

Declared navigation titles: `MuckBank` (`:475`), `Transaction` (`:728`), `Cards`
(`:1147`), `Activity` (`:1387`).

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls
`tabs,transfer,slider`; navigation `tabs-wizard`; motion `animated-balance`;
accessibility profile `B`; reset mode `relaunch`.

## System permission prompts to expect

None. The built `Info.plist` contains zero usage-description keys.

## State and reset behaviour

In-memory only, held by `MuckBankStore` (`:192`), an `ObservableObject`. Reset mode is
`relaunch` (`apps.tsv`) — relaunching the app restores initial state. No persistence,
no network.

## Source-declared accessibility landmarks

49 `accessibilityIdentifier` sites, grouped by area. Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Area | Identifiers | Lines |
|---|---|---|
| Root / tabs | `screen.bank.root`, `tab.accounts`, `tab.cards`, `tab.activity` | 356–366 |
| Accounts | `balance.animated` (default of an optional parameter at `:384`), `button.startTransfer`, `button.toggleBalanceVisibility`, `label.transferCount`, `toolbar.newTransfer`, `detail.transferButton` | 384–596 |
| Transactions | `transaction.detail.amount`, `transaction.disputeButton`, `transaction.disputeConfirmation` | 693–724 |
| Transfer wizard | `transfer.stepLabel`, `transfer.sourcePicker`, `transfer.destinationPicker`, `transfer.sameAccountError`, `transfer.amountField`, `transfer.amountSlider`, `transfer.memoField`, `transfer.validationError`, `transfer.validationOK`, `transfer.reviewAmount`, `transfer.reviewError`, `transfer.successTitle`, `transfer.backButton`, `transfer.nextButton`, `transfer.confirmButton`, `transfer.cancelButton`, `transfer.alertConfirm`, `transfer.makeAnotherButton`, `transfer.closeButton` | 825–1077 |
| Cards | `card.lostBanner`, `card.toggle.freeze`, `card.toggle.onlinePayments`, `card.toggle.atmWithdrawals`, `card.toggle.contactless`, `card.limitValue`, `card.limitSlider`, `card.reportLostButton`, `card.dialog.reportLost`, `card.dialog.orderReplacement` | 1247–1325 |
| Activity | `activity.resetButton`, `activity.alertReset` | 1383–1393 |

Unstable — composed from runtime values, not reliable landmarks:
`account.row.\(last4)` (`:465`), `transaction.row.\(merchant)` (`:618`),
`transfer.quickAmount.\(amount)` (`:939`), `card.tile.\(last4)` (`:1142`),
`card.status.\(last4)` (`:1186`).

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
