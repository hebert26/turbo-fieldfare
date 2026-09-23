import CryptoKit
import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

/// Opt-in P23 image-state coverage for one verified Qwen text pack, its
/// separately verified still-image companion, and the explicitly selected
/// local image corpus. The suite does not substitute tiny or synthetic data.
/// Feature-to-reference numerical parity is intentionally owned by the
/// independent P23 vision oracle and is not claimed by these state tests.
@Suite(.serialized)
struct QwenRealImageStateTests {
    @Test(.enabled(
        if: RealQwenImageConfiguration.isOptedIn,
        "set the explicit Qwen artifact, vision-pack, and image-corpus paths to run P23"))
    func realImageOrderBindsDistinctOwnersAndPositions() async throws {
        let fixture = try await RealQwenImageFixture.make()
        let preflight = try await fixture.session.preflightCheckpointImages(
            orderedImageIDs: fixture.configuration.corpus.orderedIDs,
            imagesByID: fixture.configuration.corpus.imagesByID,
            visionResidency: .onDemand)
        let result = try await fixture.session.generate(
            imageRequest(fixture), onEvent: { _ in })
        let snapshot = try await fixture.state.diagnosticSnapshot()

        #expect((await fixture.state.status()).committed == result.metrics)
        assertImageLineage(
            snapshot, preflight: preflight,
            label: "ordered image generation",
            expectedImageDigests: fixture.configuration.corpus.orderedImageDigests,
            expectedProcessorDigest: fixture.configuration.visionProcessorSHA256)
        #expect(result.promptTokens > 0)
        report(
            "image-order", fixture: fixture, snapshot: snapshot,
            result: result, comparison: nil)
        try await assertReset(fixture)
    }

    @Test(.enabled(
        if: RealQwenImageConfiguration.isOptedIn,
        "set the explicit Qwen artifact, vision-pack, and image-corpus paths to run P23"))
    func realImageSoftStopContinuesFromCommittedImageState() async throws {
        let fixture = try await RealQwenImageFixture.make()
        let stop = LockedFlag()
        let stopped = try await fixture.session.generate(
            imageRequest(fixture, maxNewTokens: 16),
            shouldStop: { stop.value },
            onEvent: { event in
                if case .prefill(_, let total) = event, total > 0 {
                    // Image prefill has completed and the next loop check is
                    // the deterministic soft-stop boundary.
                    stop.set()
                }
            })
        #expect(stopped.reason == .cancelled)
        #expect(stopped.acceptedGeneratedTokenIDs.isEmpty)
        #expect(stopped.metrics.pendingTokenCount == 0)
        #expect((await fixture.state.status()).committed == stopped.metrics)
        let stoppedSnapshot = try await fixture.state.diagnosticSnapshot()
        #expect(stoppedSnapshot.lineage.visibleRows > 0)
        assertOccurrenceDigests(
            stoppedSnapshot,
            expectedImageDigests: fixture.configuration.corpus.orderedImageDigests,
            expectedProcessorDigest: fixture.configuration.visionProcessorSHA256,
            label: "soft-stop retained image digests")

        let actualContinuation = try await fixture.session.generate(
            textRequest("continue after the soft-stopped image turn"),
            onEvent: { _ in })
        let actualContinuationSnapshot = try await fixture.state.diagnosticSnapshot()

        // The clean reference uses the same loaded model and image owners
        // after reset. It exercises the same committed image boundary before
        // taking the continuation turn.
        try await assertReset(fixture)
        let cleanStop = LockedFlag()
        let cleanStopped = try await fixture.session.generate(
            imageRequest(fixture, maxNewTokens: 16),
            shouldStop: { cleanStop.value },
            onEvent: { event in
                if case .prefill(_, let total) = event, total > 0 {
                    cleanStop.set()
                }
            })
        #expect(cleanStopped.reason == .cancelled)
        #expect(cleanStopped.acceptedGeneratedTokenIDs.isEmpty)
        let cleanStoppedSnapshot = try await fixture.state.diagnosticSnapshot()
        assertOccurrenceDigests(
            cleanStoppedSnapshot,
            expectedImageDigests: fixture.configuration.corpus.orderedImageDigests,
            expectedProcessorDigest: fixture.configuration.visionProcessorSHA256,
            label: "soft-stop clean image digests")
        _ = expectImageStateMatches(
            stoppedSnapshot, cleanStoppedSnapshot,
            label: "image soft-stop committed boundary")
        let cleanContinuation = try await fixture.session.generate(
            textRequest("continue after the soft-stopped image turn"),
            onEvent: { _ in })
        let cleanContinuationSnapshot = try await fixture.state.diagnosticSnapshot()
        #expect(actualContinuation.reason == cleanContinuation.reason)
        #expect(actualContinuation.acceptedGeneratedTokenIDs
            == cleanContinuation.acceptedGeneratedTokenIDs)
        let comparison = expectImageStateMatches(
            actualContinuationSnapshot, cleanContinuationSnapshot,
            label: "image soft-stop continuation clean reference")
        report(
            "image-soft-stop", fixture: fixture,
            snapshot: actualContinuationSnapshot, result: actualContinuation,
            comparison: comparison)
        try await assertReset(fixture)
    }

    @Test(.enabled(
        if: RealQwenImageConfiguration.isOptedIn,
        "set the explicit Qwen artifact, vision-pack, and image-corpus paths to run P23"))
    func realImageCancellationRollsBackLineageAndAllowsCleanRetry() async throws {
        let fixture = try await RealQwenImageFixture.make()
        let beforeStatus = await fixture.state.status()
        let before = try await fixture.state.diagnosticSnapshot()
        let gate = ImagePrefillCancellationGate()
        let outcome = ImageGenerationOutcome()
        let operation = Task {
            do {
                _ = try await fixture.session.generate(
                    imageRequest(fixture, maxNewTokens: 16),
                    shouldStop: { false }, onEvent: { event in gate.observe(event) })
                await outcome.record(.committed)
            } catch is CancellationError {
                await outcome.record(.cancelled)
            } catch {
                await outcome.record(.failed(String(describing: error)))
            }
        }

        let observedPrefill = await gate.waitForPrefill(
            timeoutNanoseconds: 600_000_000_000)
        guard observedPrefill else {
            Issue.record("image cancellation prefill gate was not reached within 600 seconds")
            operation.cancel()
            gate.release()
            if await outcome.wait(timeoutNanoseconds: 5_000_000_000) {
                #expect(await outcome.value == .cancelled)
                try await assertReset(fixture)
            } else {
                Issue.record(
                    "image cancellation task did not settle within 5 seconds after the 600-second prefill timeout")
            }
            return
        }
        // The callback is holding the first decode boundary. Cancellation is
        // installed before it is released, so this cannot race a committed
        // image turn.
        operation.cancel()
        gate.release()
        guard await outcome.wait(timeoutNanoseconds: 5_000_000_000) else {
            Issue.record(
                "image cancellation task did not settle within 5 seconds after release")
            return
        }
        guard await outcome.value == .cancelled else {
            Issue.record("cancelled image generation unexpectedly committed")
            try await assertReset(fixture)
            return
        }
        #expect(await fixture.state.status() == beforeStatus)

        let restored = try await fixture.state.diagnosticSnapshot()
        _ = expectImageStateMatches(
            restored, before, label: "image hard-cancel immediate rollback")
        #expect(restored.lineage == before.lineage)
        #expect(restored.replayGeneration == before.replayGeneration)
        #expect(restored.producerEpoch == before.producerEpoch)
        #expect(restored.lineage.allocationIDs.isEmpty)
        #expect(restored.lineage.liveOwnerRows == 0)
        #expect(restored.lineage.liveRequestedBytes == 0)
        #expect((await fixture.state.status()).activeTransaction == nil)

        let retry = try await fixture.session.generate(
            imageRequest(fixture), onEvent: { _ in })
        let retrySnapshot = try await fixture.state.diagnosticSnapshot()
        #expect(retrySnapshot.lineage.visibleRows > 0)
        assertOccurrenceDigests(
            retrySnapshot,
            expectedImageDigests: fixture.configuration.corpus.orderedImageDigests,
            expectedProcessorDigest: fixture.configuration.visionProcessorSHA256,
            label: "hard-cancel retry image digests")

        // Reuse the same loaded model, context, state, and session for the
        // clean sequential reference after a hard rollback.
        try await assertReset(fixture)
        let clean = try await fixture.session.generate(
            imageRequest(fixture), onEvent: { _ in })
        let cleanSnapshot = try await fixture.state.diagnosticSnapshot()
        #expect(retry.reason == clean.reason)
        #expect(retry.acceptedGeneratedTokenIDs == clean.acceptedGeneratedTokenIDs)
        let comparison = expectImageStateMatches(
            retrySnapshot, cleanSnapshot,
            label: "image hard-cancel clean retry")
        report(
            "image-hard-cancel", fixture: fixture, snapshot: retrySnapshot,
            result: retry, comparison: comparison)
        try await assertReset(fixture)
    }

    @Test(.enabled(
        if: RealQwenImageConfiguration.isOptedIn,
        "set the explicit Qwen artifact, vision-pack, and image-corpus paths to run P23"))
    func realImageCheckpointRebuildRestoresAndMatchesNextTurnCleanState() async throws {
        let fixture = try await RealQwenImageFixture.make()
        let image = imageRequest(fixture)
        let preflight = try await fixture.session.preflightCheckpointImages(
            orderedImageIDs: fixture.configuration.corpus.orderedIDs,
            imagesByID: fixture.configuration.corpus.imagesByID,
            visionResidency: .onDemand)
        _ = try await fixture.session.generate(image, onEvent: { _ in })
        let beforeRebuild = try await fixture.state.diagnosticSnapshot()
        let checkpointID = UUID()

        let preview = try await fixture.session.rebuildCheckpoint(
            checkpointRequest(fixture, checkpointID: checkpointID, commit: false),
            onEvent: { _ in })
        #expect(preview.committed == false)
        #expect(preview.retainedImageCount == preflight.retainedImageCount)
        #expect(preview.retainedImageRows == preflight.retainedImageRows)
        #expect(preview.retainedFeatureBytes == preflight.retainedFeatureBytes)
        let afterPreview = try await fixture.state.diagnosticSnapshot()
        _ = expectImageStateMatches(
            afterPreview, beforeRebuild,
            label: "image checkpoint dry-run rollback")
        #expect(afterPreview.lineage == beforeRebuild.lineage)
        #expect(afterPreview.replayGeneration == beforeRebuild.replayGeneration)
        #expect(afterPreview.producerEpoch == beforeRebuild.producerEpoch)
        #expect(afterPreview.lineage.allocationIDs == beforeRebuild.lineage.allocationIDs)
        #expect(afterPreview.lineage.liveOwnerRows == beforeRebuild.lineage.liveOwnerRows)
        #expect(afterPreview.lineage.liveRequestedBytes == beforeRebuild.lineage.liveRequestedBytes)
        #expect((await fixture.state.status()).activeTransaction == nil)

        let rebuilt = try await fixture.session.rebuildCheckpoint(
            checkpointRequest(fixture, checkpointID: checkpointID, commit: true),
            onEvent: { _ in })
        #expect(rebuilt.committed)
        #expect(rebuilt.retainedImageCount == preflight.retainedImageCount)
        #expect(rebuilt.retainedImageRows == preflight.retainedImageRows)
        #expect(rebuilt.retainedFeatureBytes == preflight.retainedFeatureBytes)
        let rebuiltSnapshot = try await fixture.state.diagnosticSnapshot()
        assertImageLineage(
            rebuiltSnapshot, preflight: preflight,
            label: "committed image checkpoint rebuild",
            expectedImageDigests: fixture.configuration.corpus.orderedImageDigests,
            expectedProcessorDigest: fixture.configuration.visionProcessorSHA256)
        #expect(rebuiltSnapshot.lineage.allocationIDs != beforeRebuild.lineage.allocationIDs)

        let actualResume = try await fixture.session.generate(
            QwenConversationGenerationRequest(
                turn: .checkpoint(checkpointID), systemPrompt: nil, tools: [],
                imagesByID: [:], thinking: .disabled,
                visionResidency: .defaultPolicy, config: generationConfig()),
            onEvent: { _ in })
        #expect(actualResume.promptTokens == 0)
        let actualResumeSnapshot = try await fixture.state.diagnosticSnapshot()

        let actualNext = try await fixture.session.generate(
            textRequest("continue after the image checkpoint"), onEvent: { _ in })
        let actualNextSnapshot = try await fixture.state.diagnosticSnapshot()

        // Resetting the same session supplies the sequential clean reference
        // without loading a second text model or retaining a second state.
        try await assertReset(fixture)
        let cleanCheckpointID = UUID()
        let cleanRebuild = try await fixture.session.rebuildCheckpoint(
            checkpointRequest(fixture, checkpointID: cleanCheckpointID, commit: true),
            onEvent: { _ in })
        #expect(cleanRebuild.committed)
        let cleanResume = try await fixture.session.generate(
            QwenConversationGenerationRequest(
                turn: .checkpoint(cleanCheckpointID), systemPrompt: nil, tools: [],
                imagesByID: [:], thinking: .disabled,
                visionResidency: .defaultPolicy, config: generationConfig()),
            onEvent: { _ in })
        #expect(cleanResume.promptTokens == 0)
        let cleanResumeSnapshot = try await fixture.state.diagnosticSnapshot()
        let cleanNext = try await fixture.session.generate(
            textRequest("continue after the image checkpoint"), onEvent: { _ in })
        let cleanNextSnapshot = try await fixture.state.diagnosticSnapshot()
        #expect(actualResume.reason == cleanResume.reason)
        #expect(actualResume.acceptedGeneratedTokenIDs == cleanResume.acceptedGeneratedTokenIDs)
        _ = expectImageStateMatches(
            actualResumeSnapshot, cleanResumeSnapshot,
            label: "image checkpoint resume clean reference")
        #expect(actualNext.reason == cleanNext.reason)
        #expect(actualNext.acceptedGeneratedTokenIDs == cleanNext.acceptedGeneratedTokenIDs)
        let comparison = expectImageStateMatches(
            actualNextSnapshot, cleanNextSnapshot,
            label: "image checkpoint next-turn clean reference")
        report(
            "image-checkpoint-next-turn", fixture: fixture,
            snapshot: actualNextSnapshot, result: actualNext,
            comparison: comparison)
        try await assertReset(fixture)
    }

    @Test(.enabled(
        if: RealQwenImageConfiguration.isOptedIn,
        "set the explicit Qwen artifact, vision-pack, and image-corpus paths to run P23"))
    func realImageReloadAndCheckpointRebuildMatchSequentialCleanState() async throws {
        // Each helper returns only after reset, so the first model, state, and
        // Metal context are released before the second fixture is constructed.
        let first = try await captureReloadedCheckpoint()
        let second = try await captureReloadedCheckpoint()

        #expect(first.result.reason == second.result.reason)
        #expect(first.result.acceptedGeneratedTokenIDs
            == second.result.acceptedGeneratedTokenIDs)
        #expect(first.preflight == second.preflight)
        #expect(first.snapshot.lineage.occurrences.map(\.absoluteTokenRange)
            == second.snapshot.lineage.occurrences.map(\.absoluteTokenRange))
        #expect(first.snapshot.lineage.occurrences.map(\.exactPositions)
            == second.snapshot.lineage.occurrences.map(\.exactPositions))
        #expect(first.snapshot.replayGeneration == second.snapshot.replayGeneration)
        #expect(first.snapshot.producerEpoch == second.snapshot.producerEpoch)
        _ = expectImageStateMatches(
            first.snapshot, second.snapshot,
            label: "image reload checkpoint clean reference")
        print(
            "P23 case=image-reload-checkpoint modelID=\(first.modelID) "
                + "textManifest=\(first.textManifestSHA256) "
                + "visionManifest=\(first.visionManifestSHA256) "
                + "corpus=\(first.corpusRoot) "
                + "images=\(first.imageSummary) "
                + "retainedRows=\(first.snapshot.lineage.visibleRows) "
                + "ownerBytes=\(first.snapshot.lineage.ownedRequestedBytes) "
                + "sequence=\(first.snapshot.runnerState.sequenceLength) "
                + "replayGeneration=\(first.snapshot.replayGeneration) "
                + "producerEpoch=\(first.snapshot.producerEpoch) "
                + "stateBound=2e-5+2e-5*scale")
    }

    @Test(.enabled(
        if: RealQwenImageConfiguration.isOptedIn,
        "set the explicit Qwen artifact, vision-pack, and image-corpus paths to run P23"))
    func realImageChangeReleasesOldOwnersAndMatchesChangedCleanState() async throws {
        let fixture = try await RealQwenImageFixture.make()
        let originalImages = fixture.configuration.corpus.imagesByID
        let originalPreflight = try await fixture.session.preflightCheckpointImages(
            orderedImageIDs: fixture.configuration.corpus.orderedIDs,
            imagesByID: originalImages,
            visionResidency: .onDemand)
        let originalResult = try await fixture.session.generate(
            imageRequest(fixture, imagesByID: originalImages), onEvent: { _ in })
        let originalSnapshot = try await fixture.state.diagnosticSnapshot()
        #expect((await fixture.state.status()).committed == originalResult.metrics)
        assertImageLineage(
            originalSnapshot, preflight: originalPreflight,
            label: "original image binding",
            expectedImageDigests: fixture.configuration.corpus.orderedImageDigests,
            expectedProcessorDigest: fixture.configuration.visionProcessorSHA256)

        try await assertReset(fixture)
        let released = try await fixture.state.diagnosticSnapshot()
        #expect(released.currentLogits == nil)
        #expect(released.lineage.allocationIDs.isEmpty)
        #expect(released.lineage.occurrences.isEmpty)
        #expect(released.lineage.liveOwnerRows == 0)
        #expect(released.lineage.liveRequestedBytes == 0)

        // Both URLs are the two corpus files already admitted and hashed by
        // RealQwenImageCorpus.load. Swapping them changes the ordered digest
        // binding without mutating or copying either authorized input.
        let changedFirst = try #require(originalImages["second"])
        let changedSecond = try #require(originalImages["first"])
        let changedImages = ["first": changedFirst, "second": changedSecond]
        #expect(Set(changedImages.values) == Set(originalImages.values))
        #expect(changedImages["first"] != originalImages["first"])
        let changedBinding = try imageBindingSummary(
            changedImages, corpus: fixture.configuration.corpus)
        let changedImageDigests = try admittedImageDigests(
            changedImages, corpus: fixture.configuration.corpus)
        let changedPreflight = try await fixture.session.preflightCheckpointImages(
            orderedImageIDs: fixture.configuration.corpus.orderedIDs,
            imagesByID: changedImages,
            visionResidency: .onDemand)
        let changedResult = try await fixture.session.generate(
            imageRequest(fixture, imagesByID: changedImages), onEvent: { _ in })
        let changedSnapshot = try await fixture.state.diagnosticSnapshot()
        #expect((await fixture.state.status()).committed == changedResult.metrics)
        assertImageLineage(
            changedSnapshot, preflight: changedPreflight,
            label: "changed image binding",
            expectedImageDigests: changedImageDigests,
            expectedProcessorDigest: fixture.configuration.visionProcessorSHA256)
        #expect(changedSnapshot.lineage.allocationIDs.isDisjoint(
            with: originalSnapshot.lineage.allocationIDs),
            "changed images must not reuse released owner allocations")
        #expect(changedSnapshot.lineage.occurrences.map(\.exactPositions)
            != originalSnapshot.lineage.occurrences.map(\.exactPositions),
            "changed image geometry must not retain old M-RoPE spans")
        #expect(changedSnapshot.currentLogits != originalSnapshot.currentLogits,
                "changed image must produce new logits, not stale retained features")
        #expect(changedSnapshot.lineage.visibleRows == changedPreflight.retainedImageRows)
        #expect(changedSnapshot.lineage.ownedRequestedBytes
            == changedPreflight.retainedFeatureBytes)

        try await assertReset(fixture)
        let afterChangedRelease = try await fixture.state.diagnosticSnapshot()
        #expect(afterChangedRelease.currentLogits == nil)
        #expect(afterChangedRelease.lineage.allocationIDs.isEmpty)
        #expect(afterChangedRelease.lineage.occurrences.isEmpty)
        #expect(afterChangedRelease.lineage.liveOwnerRows == 0)
        #expect(afterChangedRelease.lineage.liveRequestedBytes == 0)

        let cleanChangedResult = try await fixture.session.generate(
            imageRequest(fixture, imagesByID: changedImages), onEvent: { _ in })
        let cleanChangedSnapshot = try await fixture.state.diagnosticSnapshot()
        assertOccurrenceDigests(
            cleanChangedSnapshot,
            expectedImageDigests: changedImageDigests,
            expectedProcessorDigest: fixture.configuration.visionProcessorSHA256,
            label: "changed image clean digests")
        #expect(changedResult.reason == cleanChangedResult.reason)
        #expect(changedResult.acceptedGeneratedTokenIDs
            == cleanChangedResult.acceptedGeneratedTokenIDs)
        let comparison = expectImageStateMatches(
            changedSnapshot, cleanChangedSnapshot,
            label: "changed image clean sequential reference")
        report(
            "image-changed-release", fixture: fixture,
            snapshot: changedSnapshot, result: changedResult,
            comparison: comparison, imageSummary: changedBinding)
        try await assertReset(fixture)
    }

    private func captureReloadedCheckpoint() async throws -> ReloadedImageEvidence {
        let fixture = try await RealQwenImageFixture.make()
        let preflight = try await fixture.session.preflightCheckpointImages(
            orderedImageIDs: fixture.configuration.corpus.orderedIDs,
            imagesByID: fixture.configuration.corpus.imagesByID,
            visionResidency: .onDemand)
        let checkpointID = UUID()
        let rebuilt = try await fixture.session.rebuildCheckpoint(
            checkpointRequest(fixture, checkpointID: checkpointID, commit: true),
            onEvent: { _ in })
        #expect(rebuilt.committed)
        #expect(rebuilt.retainedImageCount == preflight.retainedImageCount)
        #expect(rebuilt.retainedImageRows == preflight.retainedImageRows)
        #expect(rebuilt.retainedFeatureBytes == preflight.retainedFeatureBytes)
        let result = try await fixture.session.generate(
            QwenConversationGenerationRequest(
                turn: .checkpoint(checkpointID), systemPrompt: nil, tools: [],
                imagesByID: [:], thinking: .disabled,
                visionResidency: .defaultPolicy, config: generationConfig()),
            onEvent: { _ in })
        let snapshot = try await fixture.state.diagnosticSnapshot()
        #expect((await fixture.state.status()).committed == result.metrics)
        assertImageLineage(
            snapshot, preflight: preflight,
            label: "reloaded image checkpoint",
            expectedImageDigests: fixture.configuration.corpus.orderedImageDigests,
            expectedProcessorDigest: fixture.configuration.visionProcessorSHA256)
        let evidence = ReloadedImageEvidence(
            result: result, snapshot: snapshot, preflight: preflight,
            modelID: fixture.configuration.identity.modelID,
            textManifestSHA256: fixture.configuration.identity.textManifestSHA256,
            visionManifestSHA256: fixture.configuration.visionManifestSHA256,
            corpusRoot: fixture.configuration.corpus.rootURL.path,
            imageSummary: fixture.configuration.corpus.imageSummary)
        try await assertReset(fixture)
        return evidence
    }

    private func imageRequest(
        _ fixture: RealQwenImageFixture,
        maxNewTokens: Int = 1,
        imagesByID: [String: URL]? = nil
    ) -> QwenConversationGenerationRequest {
        QwenConversationGenerationRequest(
            turn: .user(orderedImageMessage()), systemPrompt: nil, tools: [],
            imagesByID: imagesByID ?? fixture.configuration.corpus.imagesByID,
            thinking: .disabled, visionResidency: .onDemand,
            config: generationConfig(maxNewTokens: maxNewTokens))
    }

    private func textRequest(_ text: String) -> QwenConversationGenerationRequest {
        QwenConversationGenerationRequest(
            turn: .user(.init(role: .user, content: text)), systemPrompt: nil,
            tools: [], imagesByID: [:], thinking: .disabled,
            visionResidency: .defaultPolicy, config: generationConfig())
    }

    private func checkpointRequest(
        _ fixture: RealQwenImageFixture,
        checkpointID: UUID,
        commit: Bool
    ) -> QwenConversationCheckpointRequest {
        QwenConversationCheckpointRequest(
            checkpointID: checkpointID,
            messages: [orderedImageMessage()], tools: [],
            imagesByID: fixture.configuration.corpus.imagesByID,
            thinking: .disabled, visionResidency: .onDemand,
            reason: .capacity, commit: commit)
    }

    private func assertReset(_ fixture: RealQwenImageFixture) async throws {
        try await fixture.session.reset()
        let snapshot = try await fixture.state.diagnosticSnapshot()
        #expect(snapshot.retainedTokenIDs.isEmpty, "reset clears retained tokens")
        #expect(snapshot.consumedTokenIDs.isEmpty, "reset clears consumed tokens")
        #expect(snapshot.pendingAcceptedToken == nil, "reset clears pending token")
        #expect(snapshot.currentLogits == nil, "reset clears current logits")
        #expect(snapshot.lineage.visibleRows == 0, "reset clears visible image rows")
        #expect(snapshot.lineage.allocationIDs.isEmpty, "reset releases image owners")
        #expect(snapshot.lineage.liveOwnerRows == 0, "reset leaves no live image rows")
        #expect(snapshot.lineage.liveRequestedBytes == 0,
                "reset leaves no live image allocation bytes")
        #expect((await fixture.state.status()).activeTransaction == nil,
                "reset leaves no active transaction")
    }

    private func generationConfig(maxNewTokens: Int = 1) -> GenerationConfig {
        GenerationConfig(
            maxNewTokens: maxNewTokens, temperature: 0, topK: nil, topP: nil,
            repetitionPenalty: 1, seed: 0)
    }

    private func orderedImageMessage() -> ModelChatMessage {
        ModelChatMessage(
            role: .user,
            content: .parts([
                .text("Compare "), .image(.init(id: "first")),
                .text(" with "), .image(.init(id: "second")),
            ]))
    }

    private func assertImageLineage(
        _ snapshot: QwenConversationStateDiagnosticSnapshot,
        preflight: QwenConversationImagePreflight,
        label: String,
        expectedImageDigests: [String] = [],
        expectedProcessorDigest: String? = nil
    ) {
        #expect(snapshot.lineage.visibleRows == preflight.retainedImageRows, "(label): rows")
        #expect(snapshot.lineage.ownedRequestedBytes == preflight.retainedFeatureBytes,
                "(label): owner bytes")
        #expect(snapshot.lineage.occurrences.count == preflight.retainedImageCount,
                "(label): image occurrence count")
        let occurrences = snapshot.lineage.occurrences
        let ownerIDs = occurrences.map(\.ownerAllocationID)
        #expect(Set(ownerIDs).count == ownerIDs.count, "(label): distinct image owners")
        #expect(Set(ownerIDs) == snapshot.lineage.allocationIDs,
                "(label): owner IDs bind every occurrence")
        #expect(occurrences.map(\.absoluteTokenRange)
            == occurrences.map(\.absoluteTokenRange).sorted {
                $0.lowerBound < $1.lowerBound
            }, "(label): image occurrence order")
        #expect(zip(occurrences, occurrences.dropFirst()).allSatisfy { first, second in
            first.absoluteTokenRange.upperBound <= second.absoluteTokenRange.lowerBound
        }, "(label): image occurrence ranges do not overlap")
        #expect(occurrences.allSatisfy { !$0.exactPositions.isEmpty },
                "(label): exact M-RoPE positions are retained")
        #expect(occurrences.map(\.exactPositions).reduce(0) { $0 + $1.count }
            == preflight.retainedImageRows,
            "(label): exact position row count")
        if !expectedImageDigests.isEmpty {
            assertOccurrenceDigests(
                snapshot,
                expectedImageDigests: expectedImageDigests,
                expectedProcessorDigest: expectedProcessorDigest,
                label: label)
        }
    }

    private func assertOccurrenceDigests(
        _ snapshot: QwenConversationStateDiagnosticSnapshot,
        expectedImageDigests: [String],
        expectedProcessorDigest: String?,
        label: String
    ) {
        #expect(snapshot.lineage.occurrences.map(\.imageDigest)
            == expectedImageDigests,
            "(label): retained image digests")
        if let expectedProcessorDigest {
            #expect(snapshot.lineage.occurrences.map(\.processorDigest)
                == Array(repeating: expectedProcessorDigest,
                         count: expectedImageDigests.count),
                "(label): retained processor digests")
        }
    }

    private func report(
        _ name: String,
        fixture: RealQwenImageFixture,
        snapshot: QwenConversationStateDiagnosticSnapshot,
        result: QwenConversationGenerationResult,
        comparison: ImageStateComparison?,
        imageSummary: String? = nil
    ) {
        let occurrenceSummary = snapshot.lineage.occurrences.map {
            "range=\($0.absoluteTokenRange) positions=\($0.exactPositions.count) "
                + "imageDigest=\($0.imageDigest) processorDigest=\($0.processorDigest)"
        }.joined(separator: ",")
        let comparisonSummary = comparison.map {
            "finite=\($0.finite) maxAbsDelta=\($0.maximumAbsoluteDelta) "
                + "scale=\($0.comparisonScale)"
        } ?? "finite=none"
        print(
            "P23 case=\(name) outcome=\(result.reason) "
                + "modelID=\(fixture.configuration.identity.modelID) "
                + "revision=\(fixture.configuration.identity.sourceRevision) "
                + "textManifest=\(fixture.configuration.identity.textManifestSHA256) "
                + "visionManifest=\(fixture.configuration.visionManifestSHA256) "
                + "visionProcessor=\(fixture.configuration.visionProcessorSHA256) "
                + "visionPayload=\(fixture.configuration.visionPayloadSHA256) "
                + "artifact=\(fixture.configuration.artifactDirectory.path) "
                + "visionPack=\(fixture.configuration.visionPackURL.path) "
                + "corpus=\(fixture.configuration.corpus.rootURL.path) "
                + "corpusIndex=\(fixture.configuration.corpus.indexSHA256) "
                + "images=\(imageSummary ?? fixture.configuration.corpus.imageSummary) "
                + "retainedRows=\(snapshot.lineage.visibleRows) "
                + "ownerBytes=\(snapshot.lineage.ownedRequestedBytes) "
                + "sequence=\(snapshot.runnerState.sequenceLength) "
                + "replayGeneration=\(snapshot.replayGeneration) "
                + "producerEpoch=\(snapshot.producerEpoch) "
                + "occurrences=[\(occurrenceSummary)] "
                + "stateBound=2e-5+2e-5*scale \(comparisonSummary)")
    }
}

private struct ReloadedImageEvidence: Sendable {
    let result: QwenConversationGenerationResult
    let snapshot: QwenConversationStateDiagnosticSnapshot
    let preflight: QwenConversationImagePreflight
    let modelID: String
    let textManifestSHA256: String
    let visionManifestSHA256: String
    let corpusRoot: String
    let imageSummary: String
}

private enum ImageGenerationTerminal: Equatable, Sendable {
    case committed
    case cancelled
    case failed(String)
}

private actor ImageGenerationOutcome {
    private var terminal: ImageGenerationTerminal?

    func record(_ terminal: ImageGenerationTerminal) {
        guard self.terminal == nil else { return }
        self.terminal = terminal
    }

    var value: ImageGenerationTerminal? { terminal }

    func wait(timeoutNanoseconds: UInt64) async -> Bool {
        var elapsed: UInt64 = 0
        while elapsed < timeoutNanoseconds {
            if terminal != nil { return true }
            do {
                try await Task.sleep(nanoseconds: 10_000_000)
            } catch {
                return false
            }
            elapsed += 10_000_000
        }
        return terminal != nil
    }
}

private struct RealQwenImageConfiguration: Sendable {
    static let artifactEnvironmentKey = "TURBO_FIELDFARE_REAL_QWEN_ARTIFACT"
    static let visionEnvironmentKey = "TURBO_FIELDFARE_REAL_QWEN_VISION_PACK"
    static let corpusEnvironmentKey = "TURBO_FIELDFARE_REAL_QWEN_IMAGE_CORPUS"
    static let expectedModelID = "Qwen/Qwen3.6-35B-A3B"
    static let expectedSourceRevision =
        "995ad96eacd98c81ed38be0c5b274b04031597b0"

    let artifactDirectory: URL
    let visionPackURL: URL
    let identity: LoadedRuntimeIdentity
    let visionManifestSHA256: String
    let visionProcessorSHA256: String
    let visionPayloadSHA256: String
    let corpus: RealQwenImageCorpus
    let manifest: LoadedModelManifest

    static var isOptedIn: Bool {
        let environment = ProcessInfo.processInfo.environment
        return [artifactEnvironmentKey, visionEnvironmentKey, corpusEnvironmentKey]
            .allSatisfy { key in
                guard let value = environment[key] else { return false }
                return !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }
    }

    static func load() throws -> Self {
        let environment = ProcessInfo.processInfo.environment
        func required(_ key: String) throws -> String {
            guard let value = environment[key],
                  !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw RealQwenImageConfigurationError.missingEnvironment(key)
            }
            return value
        }

        let artifactDirectory = try regularDirectory(
            canonical(URL(fileURLWithPath: try required(artifactEnvironmentKey))))
        guard case .qwenV2(let manifest) = try ModelFamilyAdmission.classify(
            directoryURL: artifactDirectory) else {
            throw RealQwenImageConfigurationError.wrongTextFamily
        }
        let identity = LoadedRuntimeIdentity(descriptor: manifest.descriptor)
        guard identity.family == .qwen3_6,
              identity.modelID == expectedModelID,
              identity.sourceRevision == expectedSourceRevision else {
            throw RealQwenImageConfigurationError.identityMismatch(identity.modelID)
        }

        let visionPackURL = try regularDirectory(
            canonical(URL(fileURLWithPath: try required(visionEnvironmentKey))))
        let companion = try ModelFamilyGenerationSession.openVerifiedQwenVisionCompanion(
            directoryURL: artifactDirectory,
            loadedIdentity: identity,
            visionPackURL: visionPackURL)
        let visionManifest = companion.store.manifest
        guard visionManifest.modelID == expectedModelID,
              visionManifest.sourceRevision == expectedSourceRevision,
              visionManifest.compatibleTextManifestSHA256 == identity.textManifestSHA256,
              visionManifest.supportsStillImages,
              !visionManifest.supportsVideo else {
            throw RealQwenImageConfigurationError.unboundVisionCompanion
        }

        let corpus = try RealQwenImageCorpus.load(
            rootURL: regularDirectory(canonical(URL(
                fileURLWithPath: try required(corpusEnvironmentKey),
                isDirectory: true))))
        return Self(
            artifactDirectory: artifactDirectory,
            visionPackURL: visionPackURL,
            identity: identity,
            visionManifestSHA256: try sha256File(
                visionPackURL.appendingPathComponent("manifest.json")),
            visionProcessorSHA256: visionManifest.processorConfigSHA256,
            visionPayloadSHA256: visionManifest.visionPayloadSHA256,
            corpus: corpus,
            manifest: manifest)
    }
}

private enum RealQwenImageConfigurationError: Error, CustomStringConvertible {
    case missingEnvironment(String)
    case missingPath(String)
    case wrongTextFamily
    case identityMismatch(String)
    case unboundVisionCompanion
    case corpusDigestMismatch(String)

    var description: String {
        switch self {
        case .missingEnvironment(let key): "missing P23 environment value \(key)"
        case .missingPath(let path): "P23 path is not a regular directory: \(path)"
        case .wrongTextFamily: "P23 artifact was not admitted as Qwen v2"
        case .identityMismatch(let modelID): "unexpected P23 Qwen model identity \(modelID)"
        case .unboundVisionCompanion: "P23 vision companion is not still-image bound"
        case .corpusDigestMismatch(let path): "P23 corpus digest mismatch: \(path)"
        }
    }
}

private struct RealQwenImageCorpus: Sendable {
    static let indexSHA256 =
        "776bb2838e49e607eac0e65484a17c0c040d52f0ac135b3584ec41dfce074967"
    static let imageSpecs: [(id: String, file: String, sha256: String)] = [
        (
            "first", "natural-odd-4x3.jpg",
            "74de788bb77c1909b1d23fcda393e3dec0758cd4b5d1f04aff69045f1a323532"),
        (
            "second", "natural-exif-6.jpg",
            "25233efe3547a9ae99b96ce210c6dde16223dde47adfa2ca6dd211da9f6cd143"),
    ]

    let rootURL: URL
    let indexSHA256: String
    let orderedIDs: [String]
    let imagesByID: [String: URL]
    let imageDigestsByID: [String: String]
    let imageSummary: String

    var orderedImageDigests: [String] {
        orderedIDs.compactMap { imageDigestsByID[$0] }
    }

    static func load(rootURL: URL) throws -> Self {
        let indexURL = rootURL.appendingPathComponent("corpus.json")
        guard isRegularFile(indexURL) else {
            throw RealQwenImageConfigurationError.missingPath(indexURL.path)
        }
        let actualIndexSHA256 = try sha256File(indexURL)
        guard actualIndexSHA256 == Self.indexSHA256 else {
            throw RealQwenImageConfigurationError.corpusDigestMismatch(indexURL.path)
        }
        var images: [String: URL] = [:]
        var summary: [String] = []
        for spec in imageSpecs {
            let url = rootURL.appendingPathComponent(spec.file)
            guard isRegularFile(url) else {
                throw RealQwenImageConfigurationError.missingPath(url.path)
            }
            let digest = try sha256File(url)
            guard digest == spec.sha256 else {
                throw RealQwenImageConfigurationError.corpusDigestMismatch(url.path)
            }
            images[spec.id] = url
            summary.append("\(spec.id):\(spec.file):\(digest)")
        }
        return Self(
            rootURL: rootURL, indexSHA256: actualIndexSHA256,
            orderedIDs: imageSpecs.map(\.id), imagesByID: images,
            imageDigestsByID: Dictionary(uniqueKeysWithValues: imageSpecs.map {
                ($0.id, $0.sha256)
            }),
            imageSummary: summary.joined(separator: ","))
    }
}

private struct RealQwenImageFixture {
    let configuration: RealQwenImageConfiguration
    let context: MetalContext
    let model: QwenTextModel
    let state: QwenConversationState
    let session: QwenConversationGenerationSession

    static func make() async throws -> Self {
        let configuration = try RealQwenImageConfiguration.load()
        let context = try MetalContext()
        let model = try QwenTextModel.loadOfficial(
            directoryURL: configuration.artifactDirectory,
            manifest: configuration.manifest,
            device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 2_048)
        let tokenizer = try QwenTokenizer.load(from: configuration.artifactDirectory)
        let codec = QwenChatCodec(tokenizer: tokenizer)
        let session = try QwenConversationGenerationSession(
            model: model, state: state, codec: codec,
            verifiedIdentity: configuration.identity,
            modelDirectoryURL: configuration.artifactDirectory,
            context: context, maxContext: 2_048,
            visionPackURL: configuration.visionPackURL)
        return Self(
            configuration: configuration, context: context, model: model,
            state: state, session: session)
    }
}

private func canonical(_ url: URL) -> URL {
    url.standardizedFileURL.resolvingSymlinksInPath().standardizedFileURL
}

private func regularDirectory(_ url: URL) throws -> URL {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
          isDirectory.boolValue else {
        throw RealQwenImageConfigurationError.missingPath(url.path)
    }
    return url
}

private func isRegularFile(_ url: URL) -> Bool {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
          !isDirectory.boolValue else { return false }
    let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
    return values?.isRegularFile == true && values?.isSymbolicLink != true
}

private func sha256File(_ url: URL) throws -> String {
    let data = try Data(contentsOf: url, options: [.mappedIfSafe])
    return SHA256.hash(data: data)
        .map { String(format: "%02x", $0) }
        .joined()
}

private func imageBindingSummary(
    _ imagesByID: [String: URL], corpus: RealQwenImageCorpus
) throws -> String {
    try imagesByID.keys.sorted().map { id in
        let url = try #require(imagesByID[id])
        let digest = try admittedDigest(for: url, corpus: corpus)
        return "\(id):\(url.path):\(digest)"
    }.joined(separator: ",")
}

private func admittedImageDigests(
    _ imagesByID: [String: URL], corpus: RealQwenImageCorpus
) throws -> [String] {
    try corpus.orderedIDs.map { id in
        let url = try #require(imagesByID[id])
        return try admittedDigest(for: url, corpus: corpus)
    }
}

private func admittedDigest(
    for url: URL, corpus: RealQwenImageCorpus
) throws -> String {
    let canonicalURL = canonical(url)
    guard let source = corpus.imagesByID.first(where: {
        canonical($0.value) == canonicalURL
    }), let admitted = corpus.imageDigestsByID[source.key] else {
        throw RealQwenImageConfigurationError.corpusDigestMismatch(url.path)
    }
    // Revalidate the admitted input against its fixed corpus record. The
    // expected value always comes from the record, never from this read.
    guard try sha256File(url) == admitted else {
        throw RealQwenImageConfigurationError.corpusDigestMismatch(url.path)
    }
    return admitted
}

private struct StateNumerics: Sendable {
    var finite = true
    var maximumAbsoluteDelta: Float = 0
    var comparisonScale: Float = 0

    static let zero = Self()

    mutating func include(_ other: Self) {
        finite = finite && other.finite
        maximumAbsoluteDelta = max(maximumAbsoluteDelta, other.maximumAbsoluteDelta)
        comparisonScale = max(comparisonScale, other.comparisonScale)
    }
}

private struct ImageStateComparison: Sendable {
    let finite: Bool
    let maximumAbsoluteDelta: Float
    let comparisonScale: Float
}

private func expectImageStateMatches(
    _ actual: QwenConversationStateDiagnosticSnapshot,
    _ expected: QwenConversationStateDiagnosticSnapshot,
    label: String
) -> ImageStateComparison {
    #expect(actual.retainedTokenIDs == expected.retainedTokenIDs, "(label): retained tokens")
    #expect(actual.consumedTokenIDs == expected.consumedTokenIDs, "(label): consumed tokens")
    #expect(actual.pendingAcceptedToken == expected.pendingAcceptedToken, "(label): pending token")
    #expect(actual.textRoPEDelta == expected.textRoPEDelta, "(label): text RoPE delta")
    #expect(actual.logicalStateBytes == expected.logicalStateBytes, "(label): logical bytes")
    #expect(actual.lineage.visibleRows == expected.lineage.visibleRows, "(label): image rows")
    #expect(actual.lineage.ownedRequestedBytes == expected.lineage.ownedRequestedBytes,
            "(label): owner bytes")
    #expect(actual.lineage.provenanceRequestedBytes == expected.lineage.provenanceRequestedBytes,
            "(label): provenance bytes")
    #expect(actual.lineage.occurrences.map(\.absoluteTokenRange)
        == expected.lineage.occurrences.map(\.absoluteTokenRange),
        "(label): occurrence ranges")
    #expect(actual.lineage.occurrences.map(\.exactPositions)
        == expected.lineage.occurrences.map(\.exactPositions),
        "(label): exact positions")
    #expect(actual.lineage.occurrences.map(\.imageDigest)
        == expected.lineage.occurrences.map(\.imageDigest),
        "(label): image digests")
    #expect(actual.lineage.occurrences.map(\.processorDigest)
        == expected.lineage.occurrences.map(\.processorDigest),
        "(label): processor digests")
    #expect(actual.runnerState.architectureIdentity == expected.runnerState.architectureIdentity,
            "(label): architecture identity")
    #expect(actual.runnerState.sequenceLength == expected.runnerState.sequenceLength,
            "(label): sequence length")
    #expect(actual.runnerState.layers.count == expected.runnerState.layers.count,
            "(label): layer count")

    var numerics = StateNumerics.zero
    for (index, pair) in zip(actual.runnerState.layers, expected.runnerState.layers).enumerated() {
        switch pair {
        case let (.linear(actualHistory, actualRecurrent), .linear(expectedHistory, expectedRecurrent)):
            numerics.include(expectFiniteFloatArraysClose(
                actualHistory, expectedHistory,
                label: "(label): layer (index) convolution"))
            numerics.include(expectFiniteFloatArraysClose(
                actualRecurrent, expectedRecurrent,
                label: "(label): layer (index) recurrent"))
        case let (.full(actualKey, actualValue), .full(expectedKey, expectedValue)):
            numerics.include(expectFiniteFloatArraysClose(
                actualKey, expectedKey,
                label: "(label): layer (index) key"))
            numerics.include(expectFiniteFloatArraysClose(
                actualValue, expectedValue,
                label: "(label): layer (index) value"))
        default:
            Issue.record("(label): layer (index) changed state kind")
            numerics.finite = false
        }
    }

    switch (actual.currentLogits, expected.currentLogits) {
    case let (.some(actualLogits), .some(expectedLogits)):
        numerics.include(expectFiniteFloatArraysClose(
            actualLogits.map(Float.init), expectedLogits.map(Float.init),
            label: "(label): current logits"))
    case (.none, .none):
        break
    default:
        Issue.record("(label): current logits validity changed")
        numerics.finite = false
    }
    let limit = 2e-5 + 2e-5 * numerics.comparisonScale
    #expect(limit.isFinite, "(label): comparison limit is nonfinite")
    return ImageStateComparison(
        finite: numerics.finite,
        maximumAbsoluteDelta: numerics.maximumAbsoluteDelta,
        comparisonScale: numerics.comparisonScale)
}

private func expectFiniteFloatArraysClose(
    _ actual: [Float], _ expected: [Float], label: String
) -> StateNumerics {
    guard actual.count == expected.count else {
        Issue.record("(label): element count (actual.count) != (expected.count)")
        return StateNumerics(finite: false, maximumAbsoluteDelta: 0, comparisonScale: 0)
    }
    let actualFinite = actual.allSatisfy(\.isFinite)
    let expectedFinite = expected.allSatisfy(\.isFinite)
    #expect(actualFinite, "(label): actual state contains a nonfinite value")
    #expect(expectedFinite, "(label): expected state contains a nonfinite value")
    guard actualFinite, expectedFinite else {
        return StateNumerics(finite: false, maximumAbsoluteDelta: 0, comparisonScale: 0)
    }
    let scale = max(
        actual.map { abs($0) }.max() ?? 0,
        expected.map { abs($0) }.max() ?? 0)
    let maximum = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
    let finiteMetrics = scale.isFinite && maximum.isFinite
    #expect(finiteMetrics, "(label): derived state metrics are nonfinite")
    guard finiteMetrics else {
        return StateNumerics(finite: false, maximumAbsoluteDelta: 0, comparisonScale: 0)
    }
    let limit: Float = 2e-5 + 2e-5 * scale
    #expect(maximum <= limit, "(label): max FP32 delta (maximum) > (limit)")
    return StateNumerics(
        finite: maximum <= limit,
        maximumAbsoluteDelta: maximum,
        comparisonScale: scale)
}

private final class LockedFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var storage = false

    var value: Bool { lock.withLock { storage } }

    func set() { lock.withLock { storage = true } }
}

/// Holds the real session immediately after multimodal prefill and before its
/// first decode boundary. The test cancels while this gate is held, then
/// releases it and joins the task. This removes the scheduling race between a
/// completed image prefill event and task cancellation.
private final class ImagePrefillCancellationGate: @unchecked Sendable {
    private let lock = NSLock()
    private var prefillObserved = false
    private let releaseSemaphore = DispatchSemaphore(value: 0)

    func observe(_ event: QwenConversationGenerationEvent) {
        guard case .prefill(_, let total) = event, total > 0 else { return }
        lock.withLock { prefillObserved = true }
        // The owning test always releases after cancellation. The finite
        // fallback also prevents an unexpected test failure from orphaning a
        // callback thread indefinitely.
        _ = releaseSemaphore.wait(
            timeout: DispatchTime.now() + DispatchTimeInterval.seconds(5))
    }

    func waitForPrefill(timeoutNanoseconds: UInt64) async -> Bool {
        var elapsed: UInt64 = 0
        while elapsed < timeoutNanoseconds {
            if lock.withLock({ prefillObserved }) { return true }
            do {
                try await Task.sleep(nanoseconds: 10_000_000)
            } catch {
                return false
            }
            elapsed += 10_000_000
        }
        return lock.withLock { prefillObserved }
    }

    func release() {
        releaseSemaphore.signal()
    }
}
