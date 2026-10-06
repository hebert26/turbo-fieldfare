import Foundation
import Testing
@testable import TurboFieldfareAppCore
import TurboFieldfareDecodeProtocol

@Suite struct OfficialSourceResponseMatchingTests {
    @Test func staleSourceDigestLoadOrEpochCannotMatchCurrentBinding() throws {
        let source = try DecodeSourceIdentity(
            kind: .officialSafetensorsBF16V1,
            contentDigest: String(repeating: "a", count: 64))
        let changedSource = try DecodeSourceIdentity(
            kind: .officialSafetensorsBF16V1,
            contentDigest: String(repeating: "b", count: 64))
        let loadID = UUID()
        let epoch = UUID()
        func event(sourceIdentity: DecodeSourceIdentity?,
                   loadID: UUID, epoch: UUID) -> DecodeServiceEvent {
            DecodeServiceEvent(kind: .snapshot, generationID: UUID(),
                loadedFamily: .qwen3_6, loadID: loadID,
                sourceIdentity: sourceIdentity,
                sequence: 1, textDelta: "stale", tokenCount: 1,
                conversationEpoch: epoch)
        }
        func matches(_ candidate: DecodeServiceEvent) -> Bool {
            DecodeServiceInferenceClient.matches(candidate,
                family: .qwen3_6, loadID: loadID, modelIdentity: nil,
                sourceIdentity: source, expectedEpoch: epoch)
        }

        #expect(matches(event(sourceIdentity: source, loadID: loadID, epoch: epoch)))
        #expect(!matches(event(sourceIdentity: changedSource,
                               loadID: loadID, epoch: epoch)))
        #expect(!matches(event(sourceIdentity: source,
                               loadID: UUID(), epoch: epoch)))
        #expect(!matches(event(sourceIdentity: source,
                               loadID: loadID, epoch: UUID())))
        #expect(!matches(event(sourceIdentity: nil,
                               loadID: loadID, epoch: epoch)))
    }
}
