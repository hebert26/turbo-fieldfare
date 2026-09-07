# Overview — MuckTravel

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.travel` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckTravelApp.swift` (1407 lines) | file listing |

## Purpose

A mock flight-booking app with a search form, sortable flight results, an interactive seat map, a three-step checkout, and a trips list.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `TabView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Flight search | origin/destination pickers with a swap action, a round-trip toggle, date pickers, and a cabin picker | SearchScreen, SearchCriteria |
| Results and fare details | a sortable, filterable flight list with a fare-details popover | ResultsScreen, FlightCard, FareDetailsView |
| Seat selection | an interactive seat-grid map with occupied and selected states and an updating total | SeatMapScreen, SeatButton |
| Checkout | a three-step flow (passenger details, payment, review) with field validation, then an animated confirmation | PassengerDetailsScreen, PaymentScreen, ReviewScreen, ConfirmationScreen |
| My Trips | a list of booked trips with a detail screen supporting cancellation | MyTripsView, TripDetailView |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `form,seat-grid,filters`; navigation
`tabs-checkout`; motion `standard`; accessibility profile
`B`; reset mode `relaunch`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory ObservableObject TravelStore and BookingDraft, no reset hook found in this file. Reset mode is `relaunch` (`apps.tsv`).

## Source-declared accessibility landmarks

43 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `bookAnotherButton` | MuckTravelApp.swift:1252 |
| `bookingReference` | MuckTravelApp.swift:1226 |
| `cabinPicker` | MuckTravelApp.swift:590 |
| `cancelBookingButton` | MuckTravelApp.swift:1384 |
| `cancelCheckoutButton` | MuckTravelApp.swift:1022 |
| `cardCvvField` | MuckTravelApp.swift:1064 |
| `cardExpiryField` | MuckTravelApp.swift:1059 |
| `cardHolderField` | MuckTravelApp.swift:1055 |
| `cardNumberField` | MuckTravelApp.swift:1051 |
| `confirmBookingButton` | MuckTravelApp.swift:1171 |
| `confirmSpinner` | MuckTravelApp.swift:1160 |
| `contactEmailError` | MuckTravelApp.swift:995 |
| `contactEmailField` | MuckTravelApp.swift:988 |
| `continueToCheckoutButton` | MuckTravelApp.swift:846 |
| `continueToPaymentButton` | MuckTravelApp.swift:1010 |
| `continueToReviewButton` | MuckTravelApp.swift:1089 |
| `departureDatePicker` | MuckTravelApp.swift:555 |
| `destinationPicker` | MuckTravelApp.swift:529 |
| `emptyTripsView` | MuckTravelApp.swift:1283 |
| `fareDetailsPopover` | MuckTravelApp.swift:795 |
| `nonstopToggle` | MuckTravelApp.swift:643 |
| `originPicker` | MuckTravelApp.swift:522 |
| `paymentTotalLabel` | MuckTravelApp.swift:1075 |
| `returnDatePicker` | MuckTravelApp.swift:564 |
| `reviewTotalLabel` | MuckTravelApp.swift:1149 |
| `roundTripToggle` | MuckTravelApp.swift:552 |
| `sameCityWarning` | MuckTravelApp.swift:535 |
| `saveCardToggle` | MuckTravelApp.swift:1070 |
| `screen.travel.root` | MuckTravelApp.swift:474 |
| `searchButton` | MuckTravelApp.swift:609 |
| `searchTab` | MuckTravelApp.swift:467 |
| `seatInstruction` | MuckTravelApp.swift:812 |
| `seatLegend` | MuckTravelApp.swift:923 |
| `seatRunningTotal` | MuckTravelApp.swift:835 |
| `selectedSeatsLabel` | MuckTravelApp.swift:830 |
| `sortMenu` | MuckTravelApp.swift:656 |
| `successCheckmark` | MuckTravelApp.swift:1214 |
| `swapCitiesButton` | MuckTravelApp.swift:547 |
| `travelerCountLabel` | MuckTravelApp.swift:576 |
| `travelerStepper` | MuckTravelApp.swift:579 |
| `tripCancelledLabel` | MuckTravelApp.swift:1379 |
| `tripsTab` | MuckTravelApp.swift:472 |
| `viewMyTripsButton` | MuckTravelApp.swift:1246 |

Unstable — composed from runtime values, not reliable landmarks: `"checkoutProgress_\(step`, `"fareInfo_\(flight.flightNumber`, `"passengerNameError_\(index`, `"passengerName_\(index`, `"price_\(flight.flightNumber`, `"seat_\(seat`, `"selectFlight_\(flight.flightNumber`, `"tripRow_\(trip.reference`, `id`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
