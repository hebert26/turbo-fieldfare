import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct VisionCaptureReturnedIdentityTests {
    private let bundleIdentifier = "com.example.nestmind"
    private let simulatorUDID = "00000000-0000-0000-0000-000000000001"

    @Test
    func acceptsNullObservedBundleOnlyForCompleteRejectedDispatchEvidence() throws {
        try validate(Self.root(interactionEvidence: Self.interactionEvidence(observedBundle: .null)))
    }

    @Test
    func rejectsNullObservedBundleForSuccessfulDelivery() {
        let evidence = Self.interactionEvidence(
            observedBundle: .null,
            dispatchStatus: "acknowledged_by_driver",
            submissionStarted: .bool(true),
            deliveryAcknowledged: .bool(true),
            outcomeStatus: "succeeded",
            outcomeScope: "mutation",
            outcomeReason: "ACTION_APPLIED",
            actualEventRecipientObserved: true)
        expectIdentityMismatch(Self.root(interactionEvidence: evidence))
    }

    @Test
    func rejectsNullObservedBundleWhenSubmissionIsUnknown() {
        let evidence = Self.interactionEvidence(
            observedBundle: .null,
            dispatchStatus: "unknown",
            submissionStarted: .null,
            deliveryAcknowledged: .null,
            outcomeStatus: "unknown",
            outcomeScope: "dispatch",
            outcomeReason: "DISPATCH_OUTCOME_UNKNOWN",
            actualEventRecipientObserved: false)
        expectIdentityMismatch(Self.root(interactionEvidence: evidence))
    }

    @Test
    func rejectsIncompleteInteractionEvidenceCopy() {
        var incomplete = Self.interactionEvidence(observedBundle: .null)
        incomplete.removeValue(forKey: "outcome")
        let root = Self.root(interactionEvidence: Self.interactionEvidence(observedBundle: .null),
                             additional: [
                                 "payload": .object([
                                     "interaction_evidence": .object(incomplete),
                                 ]),
                             ])
        expectIdentityMismatch(root)
    }

    @Test
    func rejectsConflictingInteractionEvidenceCopy() {
        let conflicting = Self.interactionEvidence(
            observedBundle: .string("com.example.nestmind"),
            dispatchStatus: "acknowledged_by_driver",
            submissionStarted: .bool(true),
            deliveryAcknowledged: .bool(true),
            outcomeStatus: "succeeded",
            outcomeScope: "mutation",
            outcomeReason: "ACTION_APPLIED",
            actualEventRecipientObserved: true)
        let root = Self.root(interactionEvidence: Self.interactionEvidence(observedBundle: .null),
                             additional: [
                                 "payload": .object([
                                     "interaction_evidence": .object(conflicting),
                                 ]),
                             ])
        expectIdentityMismatch(root)
    }

    @Test
    func rejectsQualifyingNullCopyWhenAnotherEvidenceCopyIsDelivered() {
        let delivered = Self.interactionEvidence(
            observedBundle: .string("com.example.nestmind"),
            dispatchStatus: "acknowledged_by_driver",
            submissionStarted: .bool(true),
            deliveryAcknowledged: .bool(true),
            outcomeStatus: "succeeded",
            outcomeScope: "mutation",
            outcomeReason: "ACTION_APPLIED",
            actualEventRecipientObserved: true)
        let root = Self.root(interactionEvidence: Self.interactionEvidence(observedBundle: .null),
                             additional: [
                                 "payload": .object([
                                     "interaction_evidence": .object(delivered),
                                 ]),
                             ])
        expectIdentityMismatch(root)
    }

    @Test
    func rejectsWrongRequestedBundleIdentityInOtherwisePermittedEvidence() {
        var binding = Self.interactionEvidence(observedBundle: .null)["binding"]!.objectValue!
        binding["requested_bundle_id"] = .string("com.example.other")
        let evidence = Self.interactionEvidence(
            observedBundle: .null,
            binding: binding)
        expectIdentityMismatch(Self.root(interactionEvidence: evidence))
    }

    @Test
    func rejectsWrongUDIDInOtherwisePermittedEvidence() {
        var binding = Self.interactionEvidence(observedBundle: .null)["binding"]!.objectValue!
        binding["udid"] = .string("10000000-0000-0000-0000-000000000001")
        let evidence = Self.interactionEvidence(
            observedBundle: .null,
            binding: binding)
        expectIdentityMismatch(Self.root(interactionEvidence: evidence))
    }

    @Test
    func rejectsWrongNonNullObservedBundle() {
        let evidence = Self.interactionEvidence(observedBundle: .string("com.example.other"))
        expectIdentityMismatch(Self.root(interactionEvidence: evidence))
    }

    @Test
    func rejectsNullObservedBundleOutsideInteractionEvidenceBinding() {
        let root = Self.root(
            interactionEvidence: Self.interactionEvidence(observedBundle: .null),
            additional: ["metadata": .object(["observed_bundle_id": .null])])
        expectIdentityMismatch(root)
    }

    @Test
    func rejectsQualifyingNullCopyWhenResponseClaimsDispatchWasAttempted() {
        let root = Self.root(
            interactionEvidence: Self.interactionEvidence(observedBundle: .null),
            additional: ["dispatch_attempted": .bool(true)])
        expectIdentityMismatch(root)
    }

    private func validate(_ value: JSONValue) throws {
        try VisionCaptureToolLoop.validateReturnedIdentity(
            in: value,
            configuration: configuration,
            refusalCode: nil)
    }

    private func expectIdentityMismatch(_ value: JSONValue, sourceLocation: SourceLocation = #_sourceLocation) {
        do {
            try validate(value)
            Issue.record("returned identity was accepted", sourceLocation: sourceLocation)
        } catch let error as VisionCaptureAgentError {
            guard case .returnedIdentityMismatch(let fieldPath, let refusalCode) = error else {
                Issue.record("unexpected VisionCapture error: \(error)", sourceLocation: sourceLocation)
                return
            }
            #expect(!fieldPath.isEmpty, sourceLocation: sourceLocation)
            #expect(refusalCode == nil, sourceLocation: sourceLocation)
        } catch {
            Issue.record("unexpected error: \(error)", sourceLocation: sourceLocation)
        }
    }

    private var configuration: VisionCaptureAgentConfiguration {
        VisionCaptureAgentConfiguration(
            bundleIdentifier: bundleIdentifier,
            simulatorUDID: simulatorUDID,
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"))
    }

    private static func root(
        interactionEvidence: [String: JSONValue],
        additional: [String: JSONValue] = [:]
    ) -> JSONValue {
        var root = additional
        root["interaction_evidence"] = .object(interactionEvidence)
        return .object(root)
    }

    private static func interactionEvidence(
        observedBundle: JSONValue,
        dispatchStatus: String = "rejected_before_submission",
        submissionStarted: JSONValue = .bool(false),
        deliveryAcknowledged: JSONValue = .bool(false),
        outcomeStatus: String = "failed",
        outcomeScope: String = "dispatch",
        outcomeReason: String = "DISPATCH_REJECTED_BEFORE_SUBMISSION",
        actualEventRecipientObserved: Bool = false,
        binding: [String: JSONValue]? = nil
    ) -> [String: JSONValue] {
        let defaultBinding: [String: JSONValue] = [
            "requested_bundle_id": .string("com.example.nestmind"),
            "observed_bundle_id": observedBundle,
            "observed_pid": .null,
            "udid": .string("00000000-0000-0000-0000-000000000001"),
        ]
        return [
            "binding": .object(binding ?? defaultBinding),
            "dispatch": .object([
                "status": .string(dispatchStatus),
                "submission_started": submissionStarted,
                "delivery_acknowledged": deliveryAcknowledged,
            ]),
            "outcome": .object([
                "status": .string(outcomeStatus),
                "scope": .string(outcomeScope),
                "reason_code": .string(outcomeReason),
            ]),
            "target": .object([
                "actual_event_recipient_observed": .bool(actualEventRecipientObserved),
                "status": .string(actualEventRecipientObserved ? "observed" : "unavailable"),
                "reason_code": .string(
                    actualEventRecipientObserved ? "TARGET_OBSERVED" : "TARGET_UNAVAILABLE"),
            ]),
        ]
    }
}
