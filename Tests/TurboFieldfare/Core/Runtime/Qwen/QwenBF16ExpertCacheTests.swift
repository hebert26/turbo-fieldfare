import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat
@testable import TurboFieldfareOfficialQwenSource

@Suite(.serialized) struct QwenBF16ExpertCacheTests {
    @Test func layerZeroCrossShardPairMatchesLiteralBytesAndExactBudget() async throws {
        let literals = ExpertCacheLiterals()
        let source = try literals.makeSource()
        defer { source.remove() }
        #expect(source.gateUpShardName != source.downShardName)
        let context = try MetalContext()
        let configuration = try literals.configuration()
        let slots = 2
        let budget = pairCacheBytes(configuration: configuration, slots: slots)
        let recorder = CheckpointLog()
        let coordinator = try makeCoordinator(
            source: source, configuration: configuration, device: context.device,
            slotCount: slots, residencyBudget: budget,
            readHooks: QwenBF16ExpertReadHooks { recorder.record($0) })

        #expect(coordinator.allocatedCacheBytes == budget)
        #expect(coordinator.slotCount == slots)
        #expect(coordinator.expertCount == configuration.expertCount)

        let expert = 3
        let lease = try await coordinator.map(expertIDs: [expert])
        #expect(lease.diagnostics.requestedExpertIDs == [expert])
        #expect(lease.diagnostics.hits == 0)
        #expect(lease.diagnostics.misses == 1)
        let mapped = try #require(lease.experts.first)
        #expect(mapped.expertID == expert)
        #expect(mapped.slot >= 0 && mapped.slot < slots)

        let gateUpWords = literals.gateUpWords(for: expert)
        let downWords = literals.downWords(for: expert)
        #expect(mapped.gateUpLength == gateUpWords.count * MemoryLayout<UInt16>.stride)
        #expect(mapped.downLength == downWords.count * MemoryLayout<UInt16>.stride)
        #expect(bufferBytes(mapped.gateUp, count: mapped.gateUpLength)
            == QwenBF16TestFixture.littleEndianBytes(gateUpWords))
        #expect(bufferBytes(mapped.down, count: mapped.downLength)
            == QwenBF16TestFixture.littleEndianBytes(downWords))
        #expect(recorder.count(.after(expert: expert, stream: .gateUp)) == 1)
        #expect(recorder.count(.after(expert: expert, stream: .down)) == 1)
        #expect(recorder.count(.publish(expert: expert)) == 1)

        let hit = try await coordinator.map(expertIDs: [expert])
        #expect(hit.diagnostics.hits == 1)
        #expect(hit.diagnostics.misses == 0)
        #expect(hit.experts[0].gateUp === mapped.gateUp)
        #expect(hit.experts[0].down === mapped.down)
        #expect(recorder.count(.before(expert: expert, stream: .gateUp)) == 1)
        #expect(recorder.count(.before(expert: expert, stream: .down)) == 1)
        try hit.cancel()
        try lease.cancel()
    }

    @Test func rejectsInvalidPairGeometryHeadersBudgetsAndExpertIDs() async throws {
        let literals = ExpertCacheLiterals()
        let source = try literals.makeSource()
        defer { source.remove() }
        let context = try MetalContext()
        let configuration = try literals.configuration()
        let exactBudget = pairCacheBytes(configuration: configuration, slots: 1)

        #expect(throws: QwenBF16ExpertCacheError.invalidGeometry) {
            _ = try makeCoordinator(
                source: source, configuration: configuration, device: context.device,
                slotCount: 0, residencyBudget: exactBudget)
        }
        #expect(throws: QwenBF16ExpertCacheError.budgetExceeded) {
            _ = try makeCoordinator(
                source: source, configuration: configuration, device: context.device,
                slotCount: 1, residencyBudget: exactBudget - 1)
        }

        let duplicateCoordinator = try makeCoordinator(
            source: source, configuration: configuration, device: context.device,
            slotCount: 2,
            residencyBudget: pairCacheBytes(configuration: configuration, slots: 2))
        do {
            _ = try await duplicateCoordinator.map(expertIDs: [4, 4])
            Issue.record("duplicate expert IDs must be rejected")
        } catch let error as QwenExpertMappingError {
            #expect(error == .duplicateExpertWithinToken(4))
        }
        let coordinator = try makeCoordinator(
            source: source, configuration: configuration, device: context.device,
            slotCount: 1, residencyBudget: exactBudget)
        do {
            _ = try await coordinator.map(expertIDs: [-1])
            Issue.record("negative expert ID must be rejected")
        } catch let error as QwenExpertMappingError {
            #expect(error == .invalidExpert(-1))
        }
        do {
            _ = try await coordinator.map(expertIDs: [configuration.expertCount])
            Issue.record("out-of-range expert ID must be rejected")
        } catch let error as QwenExpertMappingError {
            #expect(error == .invalidExpert(configuration.expertCount))
        }

        let malformedSource = try literals.makeSource(gateUpShape: [9, 6, 2])
        defer { malformedSource.remove() }
        #expect(throws: QwenBF16ExpertCacheError.invalidHeader(ExpertCacheLiterals.gateUpName)) {
            _ = try makeCoordinator(
                source: malformedSource, configuration: configuration,
                device: context.device, slotCount: 1, residencyBudget: exactBudget)
        }
    }

    @Test func asymmetricReadFailuresJoinSiblingAndRetryVictimAsAWholePair() async throws {
        for failedStream in CacheTestStream.allCases {
            let literals = ExpertCacheLiterals()
            let source = try literals.makeSource()
            defer { source.remove() }
            let context = try MetalContext()
            let configuration = try literals.configuration()
            let recorder = CheckpointLog()
            let failure = FailOnceRead(expert: 1, stream: failedStream)
            let hooks = QwenBF16ExpertReadHooks { checkpoint in
                recorder.record(checkpoint)
                if failure.consumeIfMatching(checkpoint) {
                    throw InjectedExpertReadFailure(stream: failedStream)
                }
            }
            let coordinator = try makeCoordinator(
                source: source, configuration: configuration, device: context.device,
                slotCount: 1,
                residencyBudget: pairCacheBytes(configuration: configuration, slots: 1),
                readHooks: hooks)

            let original = try await coordinator.map(expertIDs: [0])
            try original.cancel()
            var sawInjectedFailure = false
            do {
                _ = try await coordinator.map(expertIDs: [1])
                Issue.record("the selected protected-read failure must escape map")
            } catch let error as InjectedExpertReadFailure {
                #expect(error.stream == failedStream)
                sawInjectedFailure = true
            } catch {
                Issue.record("expected injected read failure, got \(error)")
            }
            #expect(sawInjectedFailure)
            #expect(recorder.count(.before(expert: 1, stream: .gateUp)) == 1)
            #expect(recorder.count(.before(expert: 1, stream: .down)) == 1)
            #expect(recorder.count(.publish(expert: 1)) == 0)
            #expect(recorder.count(.after(expert: 1, stream: failedStream.other)) == 1,
                    "the real sibling protected read must settle despite the injected failure")

            let retriedCandidate = try await coordinator.map(expertIDs: [1])
            #expect(retriedCandidate.diagnostics.hits == 0)
            #expect(retriedCandidate.diagnostics.misses == 1)
            assertMappedPair(retriedCandidate.experts[0], literals: literals, expert: 1)
            try retriedCandidate.cancel()

            let retriedVictim = try await coordinator.map(expertIDs: [0])
            #expect(retriedVictim.diagnostics.hits == 0)
            #expect(retriedVictim.diagnostics.misses == 1)
            assertMappedPair(retriedVictim.experts[0], literals: literals, expert: 0)
            try retriedVictim.cancel()

            for expert in [0, 1] {
                #expect(recorder.count(.before(expert: expert, stream: .gateUp)) == 2)
                #expect(recorder.count(.before(expert: expert, stream: .down)) == 2)
                #expect(recorder.count(.after(expert: expert, stream: .gateUp)) ==
                        (expert == 0 || failedStream == .down ? 2 : 1))
                #expect(recorder.count(.after(expert: expert, stream: .down)) ==
                        (expert == 0 || failedStream == .gateUp ? 2 : 1))
            }
        }
    }

    @Test func simultaneousPairedReadsOnSharedRetainedHandleValidateConcurrently() async throws {
        let literals = ExpertCacheLiterals()
        let source = try literals.makeSource()
        defer { source.remove() }
        let context = try MetalContext()
        let configuration = try literals.configuration()
        let expert = 4
        let recorder = CheckpointLog()
        let startGate = ConcurrentReadStartGate(participants: 2)
        let hooks = QwenBF16ExpertReadHooks { checkpoint in
            recorder.record(checkpoint)
            switch checkpoint {
            case .beforeProtectedRead(_, _):
                try startGate.arriveAndWait()
            default:
                break
            }
        }
        let coordinator = try makeCoordinator(
            source: source, configuration: configuration, device: context.device,
            slotCount: 1,
            residencyBudget: pairCacheBytes(configuration: configuration, slots: 1),
            readHooks: hooks)
        defer { startGate.abort() }

        let lease = try await coordinator.map(expertIDs: [expert])
        defer { try? lease.cancel() }
        #expect(lease.diagnostics.requestedExpertIDs == [expert])
        #expect(lease.diagnostics.hits == 0)
        #expect(lease.diagnostics.misses == 1)
        #expect(startGate.arrivalCount == 2)
        #expect(startGate.releasedAllParticipants)
        #expect(recorder.count(.before(expert: expert, stream: .gateUp)) == 1)
        #expect(recorder.count(.before(expert: expert, stream: .down)) == 1)
        #expect(recorder.count(.after(expert: expert, stream: .gateUp)) == 1)
        #expect(recorder.count(.after(expert: expert, stream: .down)) == 1)
        #expect(recorder.count(.publish(expert: expert)) == 1)
        let mapped = try #require(lease.experts.first)
        assertMappedPair(mapped, literals: literals, expert: expert)
    }

    @Test func distinctMissPairsOverlapWithinBoundAndPublishExactBytes() async throws {
        let literals = ExpertCacheLiterals()
        let source = try literals.makeSource()
        defer { source.remove() }
        let context = try MetalContext()
        let configuration = try literals.configuration()
        let recorder = CheckpointLog()
        let concurrency = PairReadConcurrencyTracker(initialPairCount: 2)
        let hooks = QwenBF16ExpertReadHooks { checkpoint in
            recorder.record(checkpoint)
            try concurrency.record(checkpoint)
        }
        let slotCount = 8
        let coordinator = try makeCoordinator(
            source: source, configuration: configuration, device: context.device,
            slotCount: slotCount,
            residencyBudget: pairCacheBytes(configuration: configuration, slots: slotCount),
            readHooks: hooks)

        let lease = try await coordinator.map(expertIDs: Array(0..<slotCount))
        defer { try? lease.cancel() }
        #expect(lease.diagnostics.requestedExpertIDs == Array(0..<slotCount))
        #expect(lease.diagnostics.hits == 0)
        #expect(lease.diagnostics.misses == slotCount)
        #expect(concurrency.initialPairCountReached)
        #expect(concurrency.maxActivePairs >= 2,
                "distinct miss pairs must overlap in the bounded loader")
        #expect(concurrency.maxActivePairs <= 4,
                "the loader must cap concurrent miss pairs at four")
        for mapped in lease.experts {
            assertMappedPair(mapped, literals: literals, expert: mapped.expertID)
            #expect(recorder.count(.publish(expert: mapped.expertID)) == 1)
        }
        let events = recorder.snapshot()
        let firstPublishIndex = try #require(events.firstIndex { checkpoint in
            if case .publish(_) = checkpoint { return true }
            return false
        })
        let completedReadsBeforePublish = events[..<firstPublishIndex].filter { checkpoint in
            if case .after(_, _) = checkpoint { return true }
            return false
        }.count
        #expect(completedReadsBeforePublish == slotCount * 2,
                "all paired reads must complete before the first publication")
        #expect(recorder.publishedExperts() == Array(0..<slotCount),
                "published pairs must follow the requested expert order")
    }

    @Test func concurrentPairFailureJoinsReadsPublishesNothingAndRetriesWholePairs() async throws {
        let literals = ExpertCacheLiterals()
        let source = try literals.makeSource()
        defer { source.remove() }
        let context = try MetalContext()
        let configuration = try literals.configuration()
        let recorder = CheckpointLog()
        let firstBatch = ConcurrentReadStartGate(participants: 4)
        let firstBatchFinished = CompletionFlag()
        let failure = FailOnceRead(expert: 1, stream: .gateUp)
        let siblingAfterRead = BlockingCheckpoint()
        let hooks = QwenBF16ExpertReadHooks { checkpoint in
            recorder.record(checkpoint)
            switch checkpoint {
            case .beforeProtectedRead(_, _):
                if !firstBatchFinished.isSet {
                    try firstBatch.arriveAndWait()
                }
                if failure.consumeIfMatching(checkpoint) {
                    throw InjectedExpertReadFailure(stream: .gateUp)
                }
            case .afterProtectedRead(let expert, let stream)
                where expert == 2 && stream == .down:
                siblingAfterRead.pause()
            default:
                break
            }
        }
        let coordinator = try makeCoordinator(
            source: source, configuration: configuration, device: context.device,
            slotCount: 2,
            residencyBudget: pairCacheBytes(configuration: configuration, slots: 2),
            readHooks: hooks)

        let returned = CompletionFlag()
        let mapping = Task { () throws -> QwenBF16ExpertLease in
            defer { returned.mark() }
            return try await coordinator.map(expertIDs: [1, 2])
        }
        defer {
            siblingAfterRead.open()
            mapping.cancel()
        }

        #expect(await siblingAfterRead.waitUntilEntered(),
                "the unaffected pair must reach its protected-read completion")
        #expect(!returned.isSet,
                "a sibling read still in flight must keep the failed map joined")
        siblingAfterRead.open()

        do {
            let unexpected = try await mapping.value
            Issue.record("a failed pair batch must not return a lease")
            try? unexpected.cancel()
        } catch let error {
            #expect((error as? InjectedExpertReadFailure)
                == InjectedExpertReadFailure(stream: .gateUp))
        }
        firstBatchFinished.mark()
        #expect(firstBatch.arrivalCount == 4)
        #expect(recorder.count(.publish(expert: 1)) == 0)
        #expect(recorder.count(.publish(expert: 2)) == 0)
        #expect(recorder.count(.after(expert: 1, stream: .gateUp)) == 0)
        #expect(recorder.count(.after(expert: 1, stream: .down)) == 1)
        #expect(recorder.count(.after(expert: 2, stream: .gateUp)) == 1)
        #expect(recorder.count(.after(expert: 2, stream: .down)) == 1)

        let retry = try await coordinator.map(expertIDs: [1, 2])
        defer { try? retry.cancel() }
        #expect(retry.diagnostics.hits == 0)
        #expect(retry.diagnostics.misses == 2)
        assertMappedPair(retry.experts[0], literals: literals, expert: 1)
        assertMappedPair(retry.experts[1], literals: literals, expert: 2)
        #expect(recorder.count(.before(expert: 1, stream: .gateUp)) == 2)
        #expect(recorder.count(.before(expert: 1, stream: .down)) == 2)
        #expect(recorder.count(.before(expert: 2, stream: .gateUp)) == 2)
        #expect(recorder.count(.before(expert: 2, stream: .down)) == 2)
        #expect(recorder.count(.after(expert: 1, stream: .gateUp)) == 1)
        #expect(recorder.count(.after(expert: 1, stream: .down)) == 2)
        #expect(recorder.count(.after(expert: 2, stream: .gateUp)) == 2)
        #expect(recorder.count(.after(expert: 2, stream: .down)) == 2)
        #expect(recorder.count(.publish(expert: 1)) == 1)
        #expect(recorder.count(.publish(expert: 2)) == 1)
    }

    @Test func secondMissBatchFailurePreservesAllMissesAndRetriesInRequestOrder() async throws {
        let literals = ExpertCacheLiterals()
        let source = try literals.makeSource()
        defer { source.remove() }
        let context = try MetalContext()
        let configuration = try literals.configuration()
        let recorder = CheckpointLog()
        let firstFailure = FailOnceRead(expert: 4, stream: .gateUp)
        let secondFailure = FailOnceRead(expert: 5, stream: .down)
        let hooks = QwenBF16ExpertReadHooks { checkpoint in
            recorder.record(checkpoint)
            if firstFailure.consumeIfMatching(checkpoint) {
                throw InjectedBatchReadFailure(expert: 4, stream: .gateUp)
            }
            if secondFailure.consumeIfMatching(checkpoint) {
                throw InjectedBatchReadFailure(expert: 5, stream: .down)
            }
        }
        let slotCount = 8
        let coordinator = try makeCoordinator(
            source: source, configuration: configuration, device: context.device,
            slotCount: slotCount,
            residencyBudget: pairCacheBytes(configuration: configuration, slots: slotCount),
            readHooks: hooks)

        var caught: InjectedBatchReadFailure?
        do {
            let unexpected = try await coordinator.map(expertIDs: Array(0..<slotCount))
            Issue.record("a second-batch read failure must reject the complete plan")
            try? unexpected.cancel()
        } catch let error as InjectedBatchReadFailure {
            caught = error
        } catch {
            Issue.record("expected the earliest injected batch read failure, got \(error)")
        }
        #expect(caught == InjectedBatchReadFailure(expert: 4, stream: .gateUp))
        #expect(recorder.publishedExperts().isEmpty,
                "completed first-batch pairs must remain unpublished after a later failure")
        for expert in 0..<slotCount {
            #expect(recorder.count(.before(expert: expert, stream: .gateUp)) == 1)
            #expect(recorder.count(.before(expert: expert, stream: .down)) == 1)
            #expect(recorder.count(.publish(expert: expert)) == 0)
        }
        #expect(recorder.count(.after(expert: 4, stream: .gateUp)) == 0)
        #expect(recorder.count(.after(expert: 5, stream: .down)) == 0)
        #expect(recorder.count(.after(expert: 4, stream: .down)) == 1)
        #expect(recorder.count(.after(expert: 5, stream: .gateUp)) == 1)

        let retry = try await coordinator.map(expertIDs: Array(0..<slotCount))
        defer { try? retry.cancel() }
        #expect(retry.diagnostics.hits == 0)
        #expect(retry.diagnostics.misses == slotCount)
        for mapped in retry.experts {
            assertMappedPair(mapped, literals: literals, expert: mapped.expertID)
        }
        #expect(recorder.publishedExperts() == Array(0..<slotCount),
                "retry publication must follow the original request order")
    }

    @Test func cancellationBetweenMissBatchesPublishesNothingAndRetriesAllPairs() async throws {
        let literals = ExpertCacheLiterals()
        let source = try literals.makeSource()
        defer { source.remove() }
        let context = try MetalContext()
        let configuration = try literals.configuration()
        let recorder = CheckpointLog()
        let secondBatchGate = BlockingCheckpoint()
        let hooks = QwenBF16ExpertReadHooks { checkpoint in
            recorder.record(checkpoint)
            if case .beforeProtectedRead(let expert, let stream) = checkpoint,
               expert == 4 && stream == .gateUp {
                secondBatchGate.pause()
            }
        }
        let slotCount = 8
        let coordinator = try makeCoordinator(
            source: source, configuration: configuration, device: context.device,
            slotCount: slotCount,
            residencyBudget: pairCacheBytes(configuration: configuration, slots: slotCount),
            readHooks: hooks)
        let mapping = Task { try await coordinator.map(expertIDs: Array(0..<slotCount)) }
        defer {
            secondBatchGate.open()
            mapping.cancel()
        }

        #expect(await secondBatchGate.waitUntilEntered(),
                "cancellation must be injected at the second miss batch")
        mapping.cancel()
        secondBatchGate.open()
        do {
            let unexpected = try await mapping.value
            Issue.record("canceled miss batches must not return a lease")
            try? unexpected.cancel()
        } catch is CancellationError {
            // Expected after all reads in the canceled batch have settled.
        } catch {
            Issue.record("expected cancellation after the second batch, got \(error)")
        }
        #expect(recorder.publishedExperts().isEmpty,
                "cancellation between batches must publish no pair")

        let retry = try await coordinator.map(expertIDs: Array(0..<slotCount))
        defer { try? retry.cancel() }
        #expect(retry.diagnostics.hits == 0)
        #expect(retry.diagnostics.misses == slotCount)
        for mapped in retry.experts {
            assertMappedPair(mapped, literals: literals, expert: mapped.expertID)
        }
        #expect(recorder.publishedExperts() == Array(0..<slotCount))
    }

    @Test func failedReplacementForcesPriorVictimToRereadBothStreams() async throws {
        for failedStream in CacheTestStream.allCases {
            let literals = ExpertCacheLiterals()
            let source = try literals.makeSource()
            defer { source.remove() }
            let context = try MetalContext()
            let configuration = try literals.configuration()
            let recorder = CheckpointLog()
            let failure = FailOnceRead(expert: 1, stream: failedStream)
            let hooks = QwenBF16ExpertReadHooks { checkpoint in
                recorder.record(checkpoint)
                if failure.consumeIfMatching(checkpoint) {
                    throw InjectedExpertReadFailure(stream: failedStream)
                }
            }
            let coordinator = try makeCoordinator(
                source: source, configuration: configuration, device: context.device,
                slotCount: 1,
                residencyBudget: pairCacheBytes(configuration: configuration, slots: 1),
                readHooks: hooks)
            let victim = try await coordinator.map(expertIDs: [0])
            try victim.cancel()

            do {
                _ = try await coordinator.map(expertIDs: [1])
                Issue.record("injected second-stream failure must reject the replacement")
            } catch let error as InjectedExpertReadFailure {
                #expect(error.stream == failedStream)
            }
            #expect(recorder.count(.after(expert: 1, stream: failedStream.other)) == 1)
            #expect(recorder.count(.publish(expert: 1)) == 0)

            let reloadedVictim = try await coordinator.map(expertIDs: [0])
            #expect(reloadedVictim.diagnostics.hits == 0)
            #expect(reloadedVictim.diagnostics.misses == 1)
            assertMappedPair(reloadedVictim.experts[0], literals: literals, expert: 0)
            #expect(recorder.count(.before(expert: 0, stream: .gateUp)) == 2)
            #expect(recorder.count(.before(expert: 0, stream: .down)) == 2)
            #expect(recorder.count(.after(expert: 0, stream: .gateUp)) == 2)
            #expect(recorder.count(.after(expert: 0, stream: .down)) == 2)
            try reloadedVictim.cancel()
        }
    }

    @Test func prePublishSourceMutationRejectsPairAndRetryCannotHitOldBytes() async throws {
        let literals = ExpertCacheLiterals()
        let source = try literals.makeSource()
        defer { source.remove() }
        let context = try MetalContext()
        let configuration = try literals.configuration()
        let recorder = CheckpointLog()
        let changedIndex = 2 * literals.downSliceWordCount
        let mutation = OneShotSourceMutation(
            source: source, tensorName: ExpertCacheLiterals.downName,
            wordIndex: changedIndex, word: literals.downWords[changedIndex] ^ 0x0001)
        let hooks = QwenBF16ExpertReadHooks { checkpoint in
            recorder.record(checkpoint)
            if case .beforePairPublish(expert: 2) = checkpoint {
                try mutation.apply()
            }
        }
        let coordinator = try makeCoordinator(
            source: source, configuration: configuration, device: context.device,
            slotCount: 1,
            residencyBudget: pairCacheBytes(configuration: configuration, slots: 1),
            readHooks: hooks)

        var firstRejected = false
        do {
            let unexpected = try await coordinator.map(expertIDs: [2])
            Issue.record("mutating a shard before pair publication must reject the pair")
            try unexpected.cancel()
        } catch let error as OfficialSourceHandleError {
            if case .replaced = error { firstRejected = true }
            else { Issue.record("expected protected-source replacement, got \(error)") }
        }
        #expect(firstRejected)
        #expect(mutation.wasApplied)
        #expect(recorder.count(.after(expert: 2, stream: .gateUp)) == 1)
        #expect(recorder.count(.after(expert: 2, stream: .down)) == 1)
        #expect(recorder.count(.publish(expert: 2)) == 1)

        var retryRejected = false
        do {
            let stale = try await coordinator.map(expertIDs: [2])
            Issue.record("retry after source invalidation must not hit stale slot bytes")
            try stale.cancel()
        } catch let error as OfficialSourceHandleError {
            if case .replaced = error { retryRejected = true }
            else { Issue.record("expected retained-source rejection on retry, got \(error)") }
        }
        #expect(retryRejected)
        #expect(recorder.count(.before(expert: 2, stream: .gateUp)) == 2)
        #expect(recorder.count(.before(expert: 2, stream: .down)) == 2)
        // The unchanged gate/up shard may finish rereading before the mutated
        // down shard rejects the pair. No second down completion or publication
        // is allowed, and the retry must not return a lease.
        #expect(recorder.count(.after(expert: 2, stream: .down)) == 1)
        #expect(recorder.count(.publish(expert: 2)) == 1)
    }

    @Test func cancellationInEitherReadJoinsBothWorkersBeforeReturning() async throws {
        for canceledStream in CacheTestStream.allCases {
            let literals = ExpertCacheLiterals()
            let source = try literals.makeSource()
            defer { source.remove() }
            let context = try MetalContext()
            let configuration = try literals.configuration()
            let recorder = CheckpointLog()
            let canceledBeforeRead = BlockingCheckpoint()
            let siblingAfterRead = BlockingCheckpoint()
            let hooks = QwenBF16ExpertReadHooks { checkpoint in
                recorder.record(checkpoint)
                switch checkpoint {
                case .beforeProtectedRead(let expert, let stream)
                    where expert == 5 && canceledStream.matches(stream):
                    canceledBeforeRead.pause()
                case .afterProtectedRead(let expert, let stream)
                    where expert == 5 && canceledStream.other.matches(stream):
                    siblingAfterRead.pause()
                default:
                    break
                }
            }
            let coordinator = try makeCoordinator(
                source: source, configuration: configuration, device: context.device,
                slotCount: 1,
                residencyBudget: pairCacheBytes(configuration: configuration, slots: 1),
                readHooks: hooks)
            let mapping = Task { try await coordinator.map(expertIDs: [5]) }
            defer {
                mapping.cancel()
                canceledBeforeRead.open()
                siblingAfterRead.open()
            }
            let canceledReadEntered = await canceledBeforeRead.waitUntilEntered()
            #expect(canceledReadEntered, "canceled stream must reach its deterministic gate")
            let siblingReadEntered = await siblingAfterRead.waitUntilEntered()
            #expect(siblingReadEntered, "sibling stream must reach its deterministic gate")

            mapping.cancel()
            siblingAfterRead.open()
            let siblingReadReturned = await siblingAfterRead.waitUntilReturned()
            #expect(siblingReadReturned, "released sibling read must settle")
            canceledBeforeRead.open()
            let canceledReadReturned = await canceledBeforeRead.waitUntilReturned()
            #expect(canceledReadReturned, "released canceled read must settle")
            var sawCancellation = false
            do {
                let unexpected = try await mapping.value
                Issue.record("a canceled paired read must not return a lease")
                try unexpected.cancel()
            } catch is CancellationError {
                sawCancellation = true
            } catch {
                Issue.record("expected cancellation after both reads settled, got \(error)")
            }
            #expect(sawCancellation)
            #expect(!canceledBeforeRead.didTimeOut)
            #expect(!siblingAfterRead.didTimeOut)
            #expect(recorder.count(.before(expert: 5, stream: .gateUp)) == 1)
            #expect(recorder.count(.before(expert: 5, stream: .down)) == 1)
            #expect(recorder.count(.after(expert: 5, stream: .gateUp)) == 1)
            #expect(recorder.count(.after(expert: 5, stream: .down)) == 1)
            #expect(recorder.count(.publish(expert: 5)) == 0)

            let retry = try await coordinator.map(expertIDs: [5])
            #expect(retry.diagnostics.hits == 0)
            #expect(retry.diagnostics.misses == 1)
            assertMappedPair(retry.experts[0], literals: literals, expert: 5)
            try retry.cancel()
        }
    }

    @Test func cachedHitRejectsMutationOfItsAdmittedSource() async throws {
        let literals = ExpertCacheLiterals()
        let source = try literals.makeSource()
        defer { source.remove() }
        let context = try MetalContext()
        let configuration = try literals.configuration()
        let recorder = CheckpointLog()
        let coordinator = try makeCoordinator(
            source: source, configuration: configuration, device: context.device,
            slotCount: 1,
            residencyBudget: pairCacheBytes(configuration: configuration, slots: 1),
            readHooks: QwenBF16ExpertReadHooks { recorder.record($0) })

        let first = try await coordinator.map(expertIDs: [4])
        #expect(first.diagnostics.misses == 1)
        try first.cancel()
        let beforeGateUp = recorder.count(.before(expert: 4, stream: .gateUp))
        let beforeDown = recorder.count(.before(expert: 4, stream: .down))
        let changedIndex = 4 * literals.downSliceWordCount
        let changed = literals.downWords[changedIndex] ^ 0x0001
        try source.replaceWord(tensorName: ExpertCacheLiterals.downName,
                               index: changedIndex, with: changed)

        var rejectedAsChangedSource = false
        do {
            let stale = try await coordinator.map(expertIDs: [4])
            Issue.record("a cached pair from a changed source must not be returned")
            try stale.cancel()
        } catch let error as OfficialSourceHandleError {
            if case .replaced = error { rejectedAsChangedSource = true }
            else { Issue.record("expected source replacement rejection, got \(error)") }
        } catch {
            Issue.record("expected real protected-source rejection, got \(error)")
        }
        #expect(rejectedAsChangedSource)
        #expect(recorder.count(.before(expert: 4, stream: .gateUp)) == beforeGateUp)
        #expect(recorder.count(.before(expert: 4, stream: .down)) == beforeDown)
    }
}

@Suite(.serialized) struct QwenBF16ExpertCacheMetalTests {
    @Test func gpuTopEightAndSharedBranchMatchIndependentLiteralCPUReference() async throws {
        let literals = ExpertCacheLiterals()
        let source = try literals.makeSource()
        defer { source.remove() }
        let context = try MetalContext()
        let configuration = try literals.configuration()
        let qwen = try QwenMoE(context: context, configuration: configuration)
        let expectedRoute = independentTopEight(
            logits: ExpertCacheLiterals.routerLogits, expertCount: configuration.expertCount,
            topK: configuration.topK)
        #expect(expectedRoute.ids == [7, 1, 5, 3, 8, 0, 6, 4])
        #expect(abs(expectedRoute.weights.reduce(Float.zero, +) - 1) <= 1e-6)
        #expect(ExpertCacheLiterals.routerLogits[expectedRoute.ids.last!] -
                ExpertCacheLiterals.routerLogits[2] == 1.0)
        let independent = independentExpertMoE(
            literals: literals, expertIDs: expectedRoute.ids,
            normalizedWeights: expectedRoute.weights)
        let gpuRoute = try encodeGPURoute(
            qwen: qwen, context: context, logits: ExpertCacheLiterals.routerLogits)
        #expect(gpuRoute.diagnostics.selectedExpertIDs == [expectedRoute.ids])
        expectRouteClose(gpuRoute.diagnostics.normalizedWeights[0], expectedRoute.weights)

        let coordinator = try makeCoordinator(
            source: source, configuration: configuration, device: context.device,
            slotCount: configuration.topK,
            residencyBudget: pairCacheBytes(configuration: configuration,
                                            slots: configuration.topK))
        let lease = try await coordinator.map(expertIDs: gpuRoute.diagnostics.selectedExpertIDs[0])
        #expect(lease.diagnostics.misses == configuration.topK)
        let sharedWeights = try literals.sharedWeights(context: context, source: source)
        let sharedNames = QwenBF16SharedNames(
            gate: ExpertCacheLiterals.sharedGateName,
            up: ExpertCacheLiterals.sharedUpName,
            down: ExpertCacheLiterals.sharedDownName,
            outputGate: ExpertCacheLiterals.sharedOutputGateName)
        let hidden = try sharedBuffer(ExpertCacheLiterals.hidden, device: context.device)
        let output = try sharedBuffer([Float](repeating: -7, count: configuration.hiddenSize),
                                      device: context.device)
        let scratch = try qwen.makeScratch()
        let command = try qwen.submitExpertsBF16(
            hidden: hidden, lease: lease, routingWeights: gpuRoute.weights,
            sharedWeights: sharedWeights, sharedNames: sharedNames,
            scratch: scratch, output: output)
        _ = await command.completed()
        #expect(command.status == .completed)
        #expect(command.error == nil)
        let actualOutput = readFloats(output, count: configuration.hiddenSize)
        expectFrozenOutput(actualOutput, independent.output)
        #expect(maxDifference(independent.output, independent.routed) > 1e-3,
                "the shared branch must contribute observably to the expected output")
        #expect(maxDifference(actualOutput, independent.routed) > 1e-3)
        #expect(lease.snapshot().completed)
        #expect(lease.snapshot().succeeded == true)

        var incorrectIDs = expectedRoute.ids
        incorrectIDs.swapAt(0, 1)
        let incorrectRouteReference = independentExpertMoE(
            literals: literals, expertIDs: incorrectIDs,
            normalizedWeights: expectedRoute.weights)
        #expect(maxDifference(independent.output, incorrectRouteReference.output) > 1e-3,
                "the literal fixture must discriminate a rank-order route error")
        let incorrectRouteLease = try await coordinator.map(expertIDs: incorrectIDs)
        #expect(incorrectRouteLease.diagnostics.hits == configuration.topK)
        let incorrectRouteOutput = try sharedBuffer(
            [Float](repeating: 0, count: configuration.hiddenSize), device: context.device)
        let incorrectRouteCommand = try qwen.submitExpertsBF16(
            hidden: hidden, lease: incorrectRouteLease, routingWeights: gpuRoute.weights,
            sharedWeights: sharedWeights, sharedNames: sharedNames,
            scratch: scratch, output: incorrectRouteOutput)
        _ = await incorrectRouteCommand.completed()
        #expect(incorrectRouteCommand.status == .completed)
        expectFrozenOutput(readFloats(incorrectRouteOutput, count: configuration.hiddenSize),
                           incorrectRouteReference.output)
        #expect(maxDifference(readFloats(incorrectRouteOutput, count: configuration.hiddenSize),
                              independent.output) > 1e-3)

        let incorrectWeights = Array(expectedRoute.weights.reversed())
        let incorrectWeightReference = independentExpertMoE(
            literals: literals, expertIDs: expectedRoute.ids,
            normalizedWeights: incorrectWeights)
        #expect(maxDifference(independent.output, incorrectWeightReference.output) > 1e-3,
                "the literal fixture must discriminate a route-weight rank error")
        let incorrectWeightLease = try await coordinator.map(expertIDs: expectedRoute.ids)
        let incorrectWeightBuffer = try sharedBuffer(incorrectWeights, device: context.device)
        let incorrectWeightOutput = try sharedBuffer(
            [Float](repeating: 0, count: configuration.hiddenSize), device: context.device)
        let incorrectWeightCommand = try qwen.submitExpertsBF16(
            hidden: hidden, lease: incorrectWeightLease,
            routingWeights: incorrectWeightBuffer, sharedWeights: sharedWeights,
            sharedNames: sharedNames, scratch: scratch, output: incorrectWeightOutput)
        _ = await incorrectWeightCommand.completed()
        #expect(incorrectWeightCommand.status == .completed)
        expectFrozenOutput(readFloats(incorrectWeightOutput, count: configuration.hiddenSize),
                           incorrectWeightReference.output)
        #expect(maxDifference(readFloats(incorrectWeightOutput, count: configuration.hiddenSize),
                              independent.output) > 1e-3)
    }

    @Test func rejectsResidentSharedWeightAliasesBeforeBF16SubmissionOrEncoding() async throws {
        let literals = ExpertCacheLiterals()
        let source = try literals.makeSource()
        defer { source.remove() }
        let context = try MetalContext()
        let configuration = try literals.configuration()
        let qwen = try QwenMoE(context: context, configuration: configuration)
        let coordinator = try makeCoordinator(
            source: source, configuration: configuration, device: context.device,
            slotCount: configuration.topK,
            residencyBudget: pairCacheBytes(configuration: configuration,
                                            slots: configuration.topK))
        let lease = try await coordinator.map(expertIDs: Array(0..<configuration.topK))
        defer { try? lease.cancel() }

        let sharedWeights = try literals.sharedWeights(context: context, source: source)
        let names = sharedNames()
        let hidden = try sharedBuffer(ExpertCacheLiterals.hidden, device: context.device)
        let routingWeights = try sharedBuffer(
            [Float](repeating: 0.125, count: configuration.topK), device: context.device)
        let scratch = try qwen.makeScratch()
        let requiredOutputBytes = configuration.hiddenSize * MemoryLayout<Float>.stride
        #expect(configuration.hiddenSize == 3)
        #expect(requiredOutputBytes == 12)

        let residentChunks = sharedWeights.inspectedChunks
        let residentBytesBefore = residentChunks.map {
            bufferBytes($0.buffer, count: $0.buffer.length)
        }
        let hiddenBytesBefore = bufferBytes(hidden, count: hidden.length)
        let routeBytesBefore = bufferBytes(routingWeights, count: routingWeights.length)
        let aliasedOutput = try #require(residentChunks.first {
            $0.name == names.gate && $0.buffer.length >= requiredOutputBytes
        })
        #expect(aliasedOutput.buffer.device === context.device)
        #expect(aliasedOutput.buffer.storageMode == .shared)
        #expect(aliasedOutput.buffer.length >= requiredOutputBytes,
                "the resident BF16 chunk is large enough for the Float32 output geometry")
        let literalSharedGateBytes = QwenBF16TestFixture.littleEndianBytes(literals.sharedGateWords)
        #expect(bufferBytes(aliasedOutput.buffer, count: literalSharedGateBytes.count)
            == literalSharedGateBytes)

        #expect(!lease.snapshot().submitted)
        #expect(throws: QwenMoEError.invalidAffineBinding(
            "BF16 shared weight aliases caller buffer")) {
            _ = try qwen.submitExpertsBF16(
                hidden: hidden, lease: lease, routingWeights: routingWeights,
                sharedWeights: sharedWeights, sharedNames: names,
                scratch: scratch, output: aliasedOutput.buffer)
        }
        #expect(!lease.snapshot().submitted,
                "alias rejection must happen before the routed lease is submitted")
        #expect(bufferBytes(hidden, count: hidden.length) == hiddenBytesBefore)
        #expect(bufferBytes(routingWeights, count: routingWeights.length) == routeBytesBefore)
        for (chunk, originalBytes) in zip(residentChunks, residentBytesBefore) {
            #expect(bufferBytes(chunk.buffer, count: chunk.buffer.length) == originalBytes,
                    "resident \(chunk.name) bytes changed after rejected MoE output alias")
        }

        let command = try #require(context.queue.makeCommandBuffer())
        #expect(command.status == .notEnqueued)
        #expect(throws: QwenMoEError.invalidAffineBinding(
            "BF16 shared weight aliases caller buffer")) {
            try qwen.encodeSharedBF16(
                commandBuffer: command, hidden: hidden, weights: sharedWeights,
                names: names, scratch: scratch, output: aliasedOutput.buffer)
        }
        #expect(command.status == .notEnqueued,
                "standalone alias rejection must leave its caller-owned command uncommitted")
        #expect(bufferBytes(hidden, count: hidden.length) == hiddenBytesBefore)
        for (chunk, originalBytes) in zip(residentChunks, residentBytesBefore) {
            #expect(bufferBytes(chunk.buffer, count: chunk.buffer.length) == originalBytes,
                    "resident \(chunk.name) bytes changed after rejected standalone alias")
        }
    }

    @Test func nonfiniteRoutesWeightsAndMalformedSubmissionGeometryAreRejected() async throws {
        let literals = ExpertCacheLiterals()
        let source = try literals.makeSource()
        defer { source.remove() }
        let context = try MetalContext()
        let configuration = try literals.configuration()
        let qwen = try QwenMoE(context: context, configuration: configuration)

        var nonfiniteLogits = ExpertCacheLiterals.routerLogits
        nonfiniteLogits[4] = .nan
        let invalidRouteBuffers = try makeRoutingBuffers(
            context: context, tokenCount: 1, topK: configuration.topK)
        let invalidRouteCommand = try #require(context.queue.makeCommandBuffer())
        try qwen.encodeRouting(
            commandBuffer: invalidRouteCommand,
            logits: sharedBuffer(nonfiniteLogits, device: context.device), tokenCount: 1,
            selectedExpertIDs: invalidRouteBuffers.selectedIDs,
            normalizedWeights: invalidRouteBuffers.weights,
            status: invalidRouteBuffers.status)
        invalidRouteCommand.commit()
        _ = await invalidRouteCommand.completed()
        #expect(invalidRouteCommand.status == .completed)
        #expect(throws: QwenMoEError.routingKernelRejectedInput(token: 0)) {
            _ = try QwenMoE.readRoutingDiagnostics(
                selectedExpertIDs: invalidRouteBuffers.selectedIDs,
                normalizedWeights: invalidRouteBuffers.weights,
                status: invalidRouteBuffers.status, tokenCount: 1,
                configuration: configuration)
        }

        let route = try encodeGPURoute(
            qwen: qwen, context: context, logits: ExpertCacheLiterals.routerLogits)
        let coordinator = try makeCoordinator(
            source: source, configuration: configuration, device: context.device,
            slotCount: configuration.topK,
            residencyBudget: pairCacheBytes(configuration: configuration,
                                            slots: configuration.topK))
        let sharedWeights = try literals.sharedWeights(context: context, source: source)
        let names = sharedNames()
        let hidden = try sharedBuffer(ExpertCacheLiterals.hidden, device: context.device)
        let output = try sharedBuffer([Float](repeating: 0, count: configuration.hiddenSize),
                                      device: context.device)
        let scratch = try qwen.makeScratch()

        let nanLease = try await coordinator.map(expertIDs: route.diagnostics.selectedExpertIDs[0])
        var nanWeights = route.diagnostics.normalizedWeights[0]
        nanWeights[3] = .nan
        #expect(throws: QwenMoEError.invalidAffineBinding(
            "nonfinite/negative BF16 routing weight")) {
            _ = try qwen.submitExpertsBF16(
                hidden: hidden, lease: nanLease,
                routingWeights: sharedBuffer(nanWeights, device: context.device),
                sharedWeights: sharedWeights, sharedNames: names,
                scratch: scratch, output: output)
        }
        #expect(!nanLease.snapshot().submitted)
        try nanLease.cancel()

        let unnormalizedLease = try await coordinator.map(
            expertIDs: route.diagnostics.selectedExpertIDs[0])
        let unnormalized = route.diagnostics.normalizedWeights[0].map { $0 * 0.5 }
        #expect(throws: QwenMoEError.invalidAffineBinding("BF16 Top-8 weights not normalized")) {
            _ = try qwen.submitExpertsBF16(
                hidden: hidden, lease: unnormalizedLease,
                routingWeights: sharedBuffer(unnormalized, device: context.device),
                sharedWeights: sharedWeights, sharedNames: names,
                scratch: scratch, output: output)
        }
        #expect(!unnormalizedLease.snapshot().submitted)
        try unnormalizedLease.cancel()

        let tooFewExperts = try await coordinator.map(
            expertIDs: Array(route.diagnostics.selectedExpertIDs[0].dropLast()))
        #expect(throws: QwenMoEError.invalidCount(
            field: "BF16 routed experts", expected: configuration.topK,
            actual: configuration.topK - 1)) {
            _ = try qwen.submitExpertsBF16(
                hidden: hidden, lease: tooFewExperts, routingWeights: route.weights,
                sharedWeights: sharedWeights, sharedNames: names,
                scratch: scratch, output: output)
        }
        #expect(!tooFewExperts.snapshot().submitted)
        try tooFewExperts.cancel()

        let shortHiddenLease = try await coordinator.map(
            expertIDs: route.diagnostics.selectedExpertIDs[0])
        let shortHidden = try sharedBuffer(
            [Float(0.25), Float(0.5)], device: context.device)
        #expect(shortHidden.length == 2 * MemoryLayout<Float>.stride)
        #expect(throws: QwenMoEError.bufferTooSmall(
            name: "BF16 hidden", required: configuration.hiddenSize * MemoryLayout<Float>.stride,
            actual: shortHidden.length)) {
            _ = try qwen.submitExpertsBF16(
                hidden: shortHidden, lease: shortHiddenLease,
                routingWeights: route.weights, sharedWeights: sharedWeights,
                sharedNames: names, scratch: scratch, output: output)
        }
        #expect(!shortHiddenLease.snapshot().submitted)
        try shortHiddenLease.cancel()
    }

    @Test func submittedPairedLeaseCannotBeEvictedBeforeActualGpuCompletion() async throws {
        let literals = ExpertCacheLiterals()
        let source = try literals.makeSource()
        defer { source.remove() }
        let context = try MetalContext()
        let configuration = try literals.configuration()
        let qwen = try QwenMoE(context: context, configuration: configuration)
        let route = try encodeGPURoute(
            qwen: qwen, context: context, logits: ExpertCacheLiterals.routerLogits)
        let selectedIDs = route.diagnostics.selectedExpertIDs[0]
        let expected = independentExpertMoE(
            literals: literals, expertIDs: selectedIDs,
            normalizedWeights: route.diagnostics.normalizedWeights[0])
        let coordinator = try makeCoordinator(
            source: source, configuration: configuration, device: context.device,
            slotCount: configuration.topK,
            residencyBudget: pairCacheBytes(configuration: configuration,
                                            slots: configuration.topK))
        let lease = try await coordinator.map(expertIDs: selectedIDs)
        let pinnedBytes = lease.experts.map { mapped in
            (mapped.expertID,
             bufferBytes(mapped.gateUp, count: mapped.gateUpLength),
             bufferBytes(mapped.down, count: mapped.downLength))
        }
        let sharedWeights = try literals.sharedWeights(context: context, source: source)
        let hidden = try sharedBuffer(ExpertCacheLiterals.hidden, device: context.device)
        let output = try sharedBuffer([Float](repeating: 0, count: configuration.hiddenSize),
                                      device: context.device)
        let scratch = try qwen.makeScratch()
        let event = try #require(context.device.makeSharedEvent())
        defer { event.signaledValue = 1 }
        let blocker = try #require(context.queue.makeCommandBuffer())
        blocker.encodeWaitForEvent(event, value: 1)
        blocker.commit()

        let command = try qwen.submitExpertsBF16(
            hidden: hidden, lease: lease, routingWeights: route.weights,
            sharedWeights: sharedWeights, sharedNames: sharedNames(),
            scratch: scratch, output: output)
        try lease.cancel()
        #expect(command.status != .notEnqueued)
        #expect(command.status != .completed)
        #expect(lease.snapshot().submitted)
        #expect(lease.snapshot().canceled)
        #expect(!lease.snapshot().completed)

        do {
            _ = try await coordinator.map(expertIDs: [2])
            Issue.record("all eight slot pairs remain pinned while the submitted command waits")
        } catch let error as QwenExpertMappingError {
            #expect(error == .insufficientUnpinnedSlots)
        }
        for (mapped, original) in zip(lease.experts, pinnedBytes) {
            #expect(mapped.expertID == original.0)
            #expect(bufferBytes(mapped.gateUp, count: mapped.gateUpLength) == original.1)
            #expect(bufferBytes(mapped.down, count: mapped.downLength) == original.2)
        }

        event.signaledValue = 1
        _ = await blocker.completed()
        _ = await command.completed()
        #expect(blocker.status == .completed)
        #expect(command.status == .completed)
        #expect(command.error == nil)
        #expect(lease.snapshot().completed)
        #expect(lease.snapshot().succeeded == false)
        expectFrozenOutput(readFloats(output, count: configuration.hiddenSize), expected.output)

        let reusable = try await coordinator.map(expertIDs: [2])
        #expect(reusable.diagnostics.hits == 0)
        #expect(reusable.diagnostics.misses == 1)
        try reusable.cancel()
    }

    @Test func submittedLiveLeaseRejectsEightMissPlanWithOnlySevenUnpinnedSlots() async throws {
        let literals = ExpertCacheLiterals()
        let source = try literals.makeSource()
        defer { source.remove() }
        let context = try MetalContext()
        let configuration = try literals.configuration()
        let slotCount = 8
        let coordinator = try makeCoordinator(
            source: source, configuration: configuration, device: context.device,
            slotCount: slotCount,
            residencyBudget: pairCacheBytes(configuration: configuration, slots: slotCount))
        let lease = try await coordinator.map(expertIDs: [0])
        let event = try #require(context.device.makeSharedEvent())
        defer { event.signaledValue = 1 }
        let command = try lease.submit(on: context.queue) { command in
            command.encodeWaitForEvent(event, value: 1)
        }
        try lease.cancel()
        #expect(lease.snapshot().submitted)
        #expect(lease.snapshot().canceled)
        #expect(!lease.snapshot().completed)

        do {
            _ = try await coordinator.map(expertIDs: Array(1...8))
            Issue.record("eight misses must be rejected while one submitted slot is live")
        } catch let error as QwenExpertMappingError {
            #expect(error == .insufficientUnpinnedSlots)
        } catch {
            Issue.record("expected insufficient unpinned slots, got \(error)")
        }

        event.signaledValue = 1
        _ = await command.completed()
        #expect(command.status == .completed)
        #expect(command.error == nil)
    }
}

private func makeCoordinator(
    source: QwenBF16ExpertCacheSourceFixture,
    configuration: QwenMoEConfiguration,
    device: MTLDevice,
    slotCount: Int,
    residencyBudget: UInt64,
    readHooks: QwenBF16ExpertReadHooks = .none
) throws -> QwenBF16ExpertMappingCoordinator {
    try QwenBF16ExpertMappingCoordinator(
        source: source.handle,
        names: QwenBF16RoutedSourceNames(
            gateUpShardName: source.gateUpShardName,
            gateUpTensorName: ExpertCacheLiterals.gateUpName,
            downShardName: source.downShardName,
            downTensorName: ExpertCacheLiterals.downName),
        layer: 0,
        configuration: configuration,
        device: device,
        slotCount: slotCount,
        residencyBudget: residencyBudget,
        readHooks: readHooks)
}

private func pairCacheBytes(configuration: QwenMoEConfiguration, slots: Int) -> UInt64 {
    let hidden = UInt64(configuration.hiddenSize)
    let intermediate = UInt64(configuration.routedIntermediateSize)
    let gateUpBytes = 2 * 2 * intermediate * hidden
    let downBytes = 2 * hidden * intermediate
    return UInt64(slots) * (gateUpBytes + downBytes)
}

private func bufferBytes(_ buffer: MTLBuffer, count: Int) -> [UInt8] {
    Array(UnsafeBufferPointer(
        start: buffer.contents().assumingMemoryBound(to: UInt8.self), count: count))
}

private struct GPUSelectionBuffers {
    let selectedIDs: MTLBuffer
    let weights: MTLBuffer
    let status: MTLBuffer
}

private struct GPURouteResult {
    let weights: MTLBuffer
    let diagnostics: QwenMoERoutingDiagnostics
}

private func makeRoutingBuffers(
    context: MetalContext, tokenCount: Int, topK: Int
) throws -> GPUSelectionBuffers {
    GPUSelectionBuffers(
        selectedIDs: try sharedBuffer([UInt32](repeating: 0, count: tokenCount * topK),
                                      device: context.device),
        weights: try sharedBuffer([Float](repeating: 0, count: tokenCount * topK),
                                  device: context.device),
        status: try sharedBuffer([UInt32](repeating: 0, count: tokenCount),
                                 device: context.device))
}

private func encodeGPURoute(
    qwen: QwenMoE, context: MetalContext, logits: [Float]
) throws -> GPURouteResult {
    let buffers = try makeRoutingBuffers(context: context, tokenCount: 1,
                                         topK: qwen.configuration.topK)
    let command = try #require(context.queue.makeCommandBuffer())
    try qwen.encodeRouting(
        commandBuffer: command, logits: sharedBuffer(logits, device: context.device),
        tokenCount: 1, selectedExpertIDs: buffers.selectedIDs,
        normalizedWeights: buffers.weights, status: buffers.status)
    command.commit()
    command.waitUntilCompleted()
    #expect(command.status == .completed)
    #expect(command.error == nil)
    let diagnostics = try QwenMoE.readRoutingDiagnostics(
        selectedExpertIDs: buffers.selectedIDs, normalizedWeights: buffers.weights,
        status: buffers.status, tokenCount: 1, configuration: qwen.configuration)
    return GPURouteResult(weights: buffers.weights, diagnostics: diagnostics)
}

private func sharedBuffer<T>(_ values: [T], device: MTLDevice) throws -> MTLBuffer {
    guard !values.isEmpty,
          let buffer = device.makeBuffer(
            bytes: values, length: values.count * MemoryLayout<T>.stride,
            options: .storageModeShared) else {
        throw QwenBF16ExpertCacheTestError.bufferAllocation
    }
    return buffer
}

private func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
    Array(UnsafeBufferPointer(
        start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
}

private func sharedNames() -> QwenBF16SharedNames {
    QwenBF16SharedNames(
        gate: ExpertCacheLiterals.sharedGateName,
        up: ExpertCacheLiterals.sharedUpName,
        down: ExpertCacheLiterals.sharedDownName,
        outputGate: ExpertCacheLiterals.sharedOutputGateName)
}

private func independentTopEight(
    logits: [Float], expertCount: Int, topK: Int
) -> (ids: [Int], weights: [Float]) {
    precondition(logits.count == expertCount && topK <= expertCount)
    precondition(logits.allSatisfy(\.isFinite))
    let maximum = logits.max()!
    let exponentials = logits.map { Foundation.exp($0 - maximum) }
    let denominator = exponentials.reduce(Float.zero, +)
    let probabilities = exponentials.map { $0 / denominator }
    let ids = logits.indices.sorted { lhs, rhs in
        if logits[lhs] == logits[rhs] { return lhs < rhs }
        return logits[lhs] > logits[rhs]
    }.prefix(topK)
    let selected = Array(ids)
    let selectedMass = selected.reduce(Float.zero) { $0 + probabilities[$1] }
    return (selected, selected.map { probabilities[$0] / selectedMass })
}

private struct IndependentExpertMoEOutput {
    let routed: [Float]
    let shared: [Float]
    let output: [Float]
}

private func independentExpertMoE(
    literals: ExpertCacheLiterals,
    expertIDs: [Int],
    normalizedWeights: [Float]
) -> IndependentExpertMoEOutput {
    let hiddenSize = 3
    let intermediateSize = 2
    precondition(expertIDs.count == 8 && normalizedWeights.count == 8)
    var routed = [Float](repeating: 0, count: hiddenSize)
    for rank in expertIDs.indices {
        let expert = expertIDs[rank]
        let gateUp = literals.gateUpWords(for: expert)
        let gate = (0..<intermediateSize).map {
            independentDot(ExpertCacheLiterals.hidden, words: gateUp,
                           row: $0, columns: hiddenSize)
        }
        let up = (0..<intermediateSize).map {
            independentDot(ExpertCacheLiterals.hidden, words: gateUp,
                           row: intermediateSize + $0, columns: hiddenSize)
        }
        let activation = (0..<intermediateSize).map {
            independentSiLU(gate[$0]) * up[$0]
        }
        let down = literals.downWords(for: expert)
        for outputRow in 0..<hiddenSize {
            let projection = independentDot(activation, words: down,
                                            row: outputRow, columns: intermediateSize)
            routed[outputRow] += projection * normalizedWeights[rank]
        }
    }

    let sharedGate = (0..<intermediateSize).map {
        independentDot(ExpertCacheLiterals.hidden, words: literals.sharedGateWords,
                       row: $0, columns: hiddenSize)
    }
    let sharedUp = (0..<intermediateSize).map {
        independentDot(ExpertCacheLiterals.hidden, words: literals.sharedUpWords,
                       row: $0, columns: hiddenSize)
    }
    let sharedActivation = (0..<intermediateSize).map {
        independentSiLU(sharedGate[$0]) * sharedUp[$0]
    }
    let sharedRawDown = (0..<hiddenSize).map {
        independentDot(sharedActivation, words: literals.sharedDownWords,
                       row: $0, columns: intermediateSize)
    }
    let rawOutputGate = independentDot(ExpertCacheLiterals.hidden,
                                       words: literals.sharedOutputGateWords,
                                       row: 0, columns: hiddenSize)
    let sharedScale = independentSigmoid(rawOutputGate)
    let shared = sharedRawDown.map { $0 * sharedScale }
    let output = zip(routed, shared).map(+)
    return IndependentExpertMoEOutput(routed: routed, shared: shared, output: output)
}

private func independentDot(
    _ input: [Float], words: [UInt16], row: Int, columns: Int
) -> Float {
    precondition(input.count == columns && (row + 1) * columns <= words.count)
    var sum: Float = 0
    for column in 0..<columns {
        sum += QwenBF16TestFixture.float(words[row * columns + column]) * input[column]
    }
    return sum
}

private func independentSiLU(_ value: Float) -> Float {
    value / (1 + Foundation.exp(-value))
}

private func independentSigmoid(_ value: Float) -> Float {
    1 / (1 + Foundation.exp(-value))
}

// Frozen before any candidate GPU result: BF16 converts exactly to FP32; the
// reference uses three-term FP32 dots, eight positive weighted expert sums,
// and one stable sigmoid. These bounds cover only their rounding/exp variation.
private let expertOutputAtol: Float = 1e-7
private let expertOutputRtol: Float = 1e-5
private let expertRouteAtol: Float = 2e-6
private let expertRouteRtol: Float = 2e-6

private func expectFrozenOutput(_ actual: [Float], _ expected: [Float]) {
    #expect(actual.count == expected.count)
    for (index, values) in zip(actual, expected).enumerated() {
        let bound = expertOutputAtol + expertOutputRtol * abs(values.1)
        #expect(values.0.isFinite && values.1.isFinite)
        #expect(abs(values.0 - values.1) <= bound,
                "BF16 MoE output[\(index)]: \(values.0) != \(values.1), bound \(bound)")
    }
}

private func expectRouteClose(_ actual: [Float], _ expected: [Float]) {
    #expect(actual.count == expected.count)
    for (actual, expected) in zip(actual, expected) {
        let bound = expertRouteAtol + expertRouteRtol * abs(expected)
        #expect(actual.isFinite && expected.isFinite)
        #expect(abs(actual - expected) <= bound)
    }
}

private func maxDifference(_ lhs: [Float], _ rhs: [Float]) -> Float {
    precondition(lhs.count == rhs.count)
    return zip(lhs, rhs).map { pair in abs(pair.0 - pair.1) }.max() ?? 0
}

private enum QwenBF16ExpertCacheTestError: Error {
    case bufferAllocation
}

private struct ExpertCacheLiterals {
    static let gateUpName = "layer.0.mlp.experts.gate_up"
    static let downName = "layer.0.mlp.experts.down"
    static let sharedGateName = "layer.0.mlp.shared.gate"
    static let sharedUpName = "layer.0.mlp.shared.up"
    static let sharedDownName = "layer.0.mlp.shared.down"
    static let sharedOutputGateName = "layer.0.mlp.shared.output_gate"

    static let routerLogits: [Float] = [
        1.0, 3.0, -2.0, 2.0, -1.0, 2.5, -0.5, 3.5, 1.5,
    ]
    static let hidden: [Float] = [0.25, 0.5, 0.75]

    // Sixteen-bit BF16 encodings are independent test literals, expert-major and
    // row-major. Each expert is [gate rows, up rows], each row has hidden=3.
    let gateUpMatrices: [[UInt16]] = [
        [0x3e80, 0x3f00, 0x3f40, 0x3f00, 0x3f40, 0x3f80,
         0x3f80, 0x3f00, 0x3e80, 0x3f40, 0x3f80, 0x3f00],
        [0x3f00, 0x3e80, 0x3f00, 0x3f40, 0x3f00, 0x3e80,
         0x3f40, 0x3f80, 0x3f00, 0x3f80, 0x3f40, 0x3f00],
        [0x3f40, 0x3f00, 0x3e80, 0x3f80, 0x3f40, 0x3f00,
         0x3f00, 0x3e80, 0x3f80, 0x3f40, 0x3f00, 0x3f80],
        [0x3e80, 0x3e00, 0x3f00, 0x3f00, 0x3f80, 0x3f40,
         0x3f80, 0x3f40, 0x3f00, 0x3f00, 0x3f00, 0x3f40],
        [0x3f00, 0x3f40, 0x3e80, 0x3f40, 0x3f80, 0x3f00,
         0x3f40, 0x3f00, 0x3f80, 0x3f80, 0x3f40, 0x3f00],
        [0x3f80, 0x3f00, 0x3f40, 0x3f40, 0x3e80, 0x3f80,
         0x3e80, 0x3f40, 0x3f00, 0x3f00, 0x3f80, 0x3f40],
        [0x3e00, 0x3e80, 0x3f00, 0x3f40, 0x3f00, 0x3f80,
         0x3f00, 0x3f80, 0x3f40, 0x3f80, 0x3f00, 0x3e80],
        [0x3f40, 0x3f80, 0x3f00, 0x3e80, 0x3f00, 0x3f40,
         0x3f80, 0x3f40, 0x3e80, 0x3f00, 0x3f40, 0x3f80],
        [0x3f00, 0x3f80, 0x3f40, 0x3f80, 0x3f00, 0x3e80,
         0x3f40, 0x3f00, 0x3f80, 0x3f40, 0x3f80, 0x3f00],
    ]

    // Each expert's down matrix is [hidden=3, intermediate=2], row-major.
    let downMatrices: [[UInt16]] = [
        [0x3e80, 0x3f00, 0x3f00, 0x3f40, 0x3f40, 0x3f80],
        [0x3f00, 0x3f40, 0x3f40, 0x3f80, 0x3f80, 0x4000],
        [0x3f40, 0x3f80, 0x3f80, 0x4000, 0x4000, 0x4000],
        [0x3e80, 0x3f40, 0x3f00, 0x3f80, 0x3f40, 0x4000],
        [0x3f00, 0x3f80, 0x3f40, 0x4000, 0x3f80, 0x4000],
        [0x3f40, 0x4000, 0x3f80, 0x4000, 0x4000, 0x4000],
        [0x3e00, 0x3e80, 0x3e80, 0x3f00, 0x3f00, 0x3f40],
        [0x3f80, 0x4000, 0x4000, 0x4040, 0x3f40, 0x3f80],
        [0x3f40, 0x3f80, 0x3f00, 0x3f40, 0x3f80, 0x4000],
    ]

    let sharedGateWords: [UInt16] = [
        0x3e80, 0x3f00, 0x3f40, 0x3f00, 0x3f40, 0x3f80,
    ]
    let sharedUpWords: [UInt16] = [
        0x3f00, 0x3f40, 0x3f80, 0x3f40, 0x3f80, 0x4000,
    ]
    let sharedDownWords: [UInt16] = [
        0x3f00, 0x3f40, 0x3f40, 0x3f80, 0x3f80, 0x4000,
    ]
    let sharedOutputGateWords: [UInt16] = [0x3f80, 0x3f00, 0x3e80]

    var gateUpWords: [UInt16] { gateUpMatrices.flatMap { $0 } }
    var downWords: [UInt16] { downMatrices.flatMap { $0 } }
    var gateUpSliceWordCount: Int { 2 * 2 * 3 }
    var downSliceWordCount: Int { 3 * 2 }

    func gateUpWords(for expert: Int) -> [UInt16] {
        let start = expert * gateUpSliceWordCount
        return Array(gateUpWords[start..<(start + gateUpSliceWordCount)])
    }

    func downWords(for expert: Int) -> [UInt16] {
        let start = expert * downSliceWordCount
        return Array(downWords[start..<(start + downSliceWordCount)])
    }

    func configuration() throws -> QwenMoEConfiguration {
        try QwenMoEConfiguration(hiddenSize: 3, expertCount: 9,
                                 routedIntermediateSize: 2,
                                 sharedIntermediateSize: 2)
    }

    func makeSource(gateUpShape: [Int] = [9, 4, 3]) throws
        -> QwenBF16ExpertCacheSourceFixture {
        try QwenBF16ExpertCacheSourceFixture.make(
            firstShard: [
                QwenBF16ExpertCacheLiteralTensor(
                    name: Self.gateUpName, shape: gateUpShape, words: gateUpWords),
                QwenBF16ExpertCacheLiteralTensor(
                    name: Self.sharedGateName, shape: [2, 3], words: sharedGateWords),
                QwenBF16ExpertCacheLiteralTensor(
                    name: Self.sharedUpName, shape: [2, 3], words: sharedUpWords),
            ],
            secondShard: [
                QwenBF16ExpertCacheLiteralTensor(
                    name: Self.downName, shape: [9, 3, 2], words: downWords),
                QwenBF16ExpertCacheLiteralTensor(
                    name: Self.sharedDownName, shape: [3, 2], words: sharedDownWords),
                QwenBF16ExpertCacheLiteralTensor(
                    name: Self.sharedOutputGateName, shape: [1, 3],
                    words: sharedOutputGateWords),
            ])
    }

    func sharedWeights(context: MetalContext,
                       source: QwenBF16ExpertCacheSourceFixture) throws -> QwenBF16Weights {
        let specifications = [
            QwenBF16TensorSpec(name: Self.sharedGateName,
                               shardName: source.gateUpShardName,
                               role: .sharedGate, rows: 2, columns: 3),
            QwenBF16TensorSpec(name: Self.sharedUpName,
                               shardName: source.gateUpShardName,
                               role: .sharedUp, rows: 2, columns: 3),
            QwenBF16TensorSpec(name: Self.sharedDownName,
                               shardName: source.downShardName,
                               role: .sharedDown, rows: 3, columns: 2),
            QwenBF16TensorSpec(name: Self.sharedOutputGateName,
                               shardName: source.downShardName,
                               role: .sharedOutputGate, rows: 1, columns: 3),
        ]
        return try QwenBF16Weights(
            context: context, source: source.handle, specifications: specifications,
            residencyBudget: UInt64((sharedGateWords.count + sharedUpWords.count
                + sharedDownWords.count + sharedOutputGateWords.count) * 2))
    }
}

private enum CacheTestStream: CaseIterable, Sendable, Equatable {
    case gateUp
    case down

    var other: Self { self == .gateUp ? .down : .gateUp }

    func matches(_ stream: QwenBF16ExpertReadHooks.Stream) -> Bool {
        switch stream {
        case .gateUp: return self == .gateUp
        case .down: return self == .down
        }
    }
}

private enum ObservedCheckpoint: Hashable, Sendable {
    case before(expert: Int, stream: CacheTestStream)
    case after(expert: Int, stream: CacheTestStream)
    case publish(expert: Int)
}

private func cacheTestStream(_ stream: QwenBF16ExpertReadHooks.Stream) -> CacheTestStream {
    switch stream {
    case .gateUp: return .gateUp
    case .down: return .down
    }
}

private final class CheckpointLog: @unchecked Sendable {
    private let lock = NSLock()
    private var entries: [ObservedCheckpoint] = []

    func record(_ checkpoint: QwenBF16ExpertReadHooks.Checkpoint) {
        let entry: ObservedCheckpoint
        switch checkpoint {
        case .beforeProtectedRead(let expert, let stream):
            entry = .before(expert: expert, stream: cacheTestStream(stream))
        case .afterProtectedRead(let expert, let stream):
            entry = .after(expert: expert, stream: cacheTestStream(stream))
        case .beforePairPublish(let expert):
            entry = .publish(expert: expert)
        }
        lock.lock()
        entries.append(entry)
        lock.unlock()
    }

    func count(_ checkpoint: ObservedCheckpoint) -> Int {
        lock.lock()
        defer { lock.unlock() }
        return entries.filter { $0 == checkpoint }.count
    }

    func snapshot() -> [ObservedCheckpoint] {
        lock.lock()
        defer { lock.unlock() }
        return entries
    }

    func publishedExperts() -> [Int] {
        snapshot().compactMap { checkpoint in
            guard case let .publish(expert) = checkpoint else { return nil }
            return expert
        }
    }
}

private struct InjectedExpertReadFailure: Error, Equatable, Sendable {
    let stream: CacheTestStream
}

private struct InjectedBatchReadFailure: Error, Equatable, Sendable {
    let expert: Int
    let stream: CacheTestStream
}

private final class FailOnceRead: @unchecked Sendable {
    private let lock = NSLock()
    private let expert: Int
    private let stream: CacheTestStream
    private var consumed = false

    init(expert: Int, stream: CacheTestStream) {
        self.expert = expert
        self.stream = stream
    }

    func consumeIfMatching(_ checkpoint: QwenBF16ExpertReadHooks.Checkpoint) -> Bool {
        guard case .beforeProtectedRead(let candidate, let candidateStream) = checkpoint,
              candidate == expert, stream.matches(candidateStream) else { return false }
        lock.lock()
        defer { lock.unlock() }
        guard !consumed else { return false }
        consumed = true
        return true
    }
}

private final class OneShotSourceMutation: @unchecked Sendable {
    private let lock = NSLock()
    private let source: QwenBF16ExpertCacheSourceFixture
    private let tensorName: String
    private let wordIndex: Int
    private let word: UInt16
    private var applied = false

    init(source: QwenBF16ExpertCacheSourceFixture, tensorName: String,
         wordIndex: Int, word: UInt16) {
        self.source = source
        self.tensorName = tensorName
        self.wordIndex = wordIndex
        self.word = word
    }

    var wasApplied: Bool {
        lock.lock()
        defer { lock.unlock() }
        return applied
    }

    func apply() throws {
        lock.lock()
        guard !applied else { lock.unlock(); return }
        applied = true
        lock.unlock()
        try source.replaceWord(tensorName: tensorName, index: wordIndex, with: word)
    }
}

private final class BlockingCheckpoint: @unchecked Sendable {
    private let lock = NSLock()
    private let entered = DispatchSemaphore(value: 0)
    private let release = DispatchSemaphore(value: 0)
    private let returned = DispatchSemaphore(value: 0)
    private var didPause = false
    private var didOpen = false
    private var timedOut = false

    var didTimeOut: Bool {
        lock.lock()
        defer { lock.unlock() }
        return timedOut
    }

    // Latch the first pause only. Repeated hook calls (notably the retry after
    // cancellation) proceed immediately rather than consuming a one-shot signal.
    func pause() {
        lock.lock()
        guard !didPause else {
            lock.unlock()
            return
        }
        didPause = true
        let alreadyOpen = didOpen
        lock.unlock()

        entered.signal()
        if !alreadyOpen && release.wait(timeout: .now() + .seconds(20)) == .timedOut {
            lock.lock()
            timedOut = true
            didOpen = true
            lock.unlock()
        }
        returned.signal()
    }

    func waitUntilEntered(timeout: DispatchTimeInterval = .seconds(8)) async -> Bool {
        await waitFor(entered, timeout: timeout)
    }

    func open() {
        lock.lock()
        let shouldSignal = !didOpen
        didOpen = true
        lock.unlock()
        if shouldSignal { release.signal() }
    }

    func waitUntilReturned(timeout: DispatchTimeInterval = .seconds(8)) async -> Bool {
        await waitFor(returned, timeout: timeout)
    }

    private func waitFor(
        _ semaphore: DispatchSemaphore, timeout: DispatchTimeInterval
    ) async -> Bool {
        let deadline = DispatchTime.now() + timeout
        return await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                continuation.resume(returning: semaphore.wait(timeout: deadline) == .success)
            }
        }
    }
}

private struct PairReadConcurrencyTimeout: Error, Sendable {}

private final class PairReadConcurrencyTracker: @unchecked Sendable {
    private let condition = NSCondition()
    private let initialPairCount: Int
    private var initialPairs: Set<Int> = []
    private var activeStreams: [Int: Int] = [:]
    private var released = false
    private var timedOut = false
    private var maximum = 0

    init(initialPairCount: Int) {
        precondition(initialPairCount > 0)
        self.initialPairCount = initialPairCount
    }

    var initialPairCountReached: Bool {
        condition.lock()
        defer { condition.unlock() }
        return initialPairs.count >= initialPairCount
    }

    var maxActivePairs: Int {
        condition.lock()
        defer { condition.unlock() }
        return maximum
    }

    func record(_ checkpoint: QwenBF16ExpertReadHooks.Checkpoint) throws {
        switch checkpoint {
        case .beforeProtectedRead(let expert, _):
            condition.lock()
            activeStreams[expert, default: 0] += 1
            initialPairs.insert(expert)
            if initialPairs.count >= initialPairCount {
                released = true
                condition.broadcast()
            }
            if !released {
                let deadline = Date(timeIntervalSinceNow: 8)
                while !released {
                    if !condition.wait(until: deadline) {
                        timedOut = true
                        released = true
                        condition.broadcast()
                    }
                }
            }
            maximum = max(maximum, activeStreams.count)
            let failed = timedOut
            condition.unlock()
            if failed { throw PairReadConcurrencyTimeout() }
        case .afterProtectedRead(let expert, _):
            condition.lock()
            if let streams = activeStreams[expert], streams > 1 {
                activeStreams[expert] = streams - 1
            } else {
                activeStreams[expert] = nil
            }
            condition.unlock()
        case .beforePairPublish(_):
            break
        }
    }
}

private final class CompletionFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false

    var isSet: Bool {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func mark() {
        lock.lock()
        value = true
        lock.unlock()
    }
}

private enum ConcurrentReadStartGateError: Error, Sendable, CustomStringConvertible {
    case timedOut(expected: Int, arrived: Int)
    case unexpectedArrivalCount(expected: Int, arrived: Int)
    case aborted

    var description: String {
        switch self {
        case .timedOut(let expected, let arrived):
            return "start barrier timed out: expected \(expected), arrived \(arrived)"
        case .unexpectedArrivalCount(let expected, let arrived):
            return "start barrier received \(arrived) arrivals, expected \(expected)"
        case .aborted:
            return "start barrier was aborted during test cleanup"
        }
    }
}

private final class ConcurrentReadStartGate: @unchecked Sendable {
    private let condition = NSCondition()
    private let expected: Int
    private let timeout: TimeInterval
    private var arrived = 0
    private var released = false
    private var failure: ConcurrentReadStartGateError?

    init(participants: Int, timeout: TimeInterval = 8) {
        precondition(participants > 0)
        self.expected = participants
        self.timeout = timeout
    }

    var arrivalCount: Int {
        condition.lock()
        defer { condition.unlock() }
        return arrived
    }

    var releasedAllParticipants: Bool {
        condition.lock()
        defer { condition.unlock() }
        return released && failure == nil && arrived == expected
    }

    func arriveAndWait() throws {
        condition.lock()
        arrived += 1
        if failure == nil {
            if arrived > expected {
                failure = .unexpectedArrivalCount(expected: expected, arrived: arrived)
                released = true
                condition.broadcast()
            } else if arrived == expected {
                released = true
                condition.broadcast()
            }
        }

        if !released {
            let deadline = Date(timeIntervalSinceNow: timeout)
            while !released {
                if !condition.wait(until: deadline) && !released {
                    failure = .timedOut(expected: expected, arrived: arrived)
                    released = true
                    condition.broadcast()
                }
            }
        }
        let result = failure
        condition.unlock()
        if let result { throw result }
    }

    func abort() {
        condition.lock()
        if !released {
            failure = .aborted
            released = true
            condition.broadcast()
        }
        condition.unlock()
    }
}

private func assertMappedPair(
    _ mapped: QwenBF16MappedExpert, literals: ExpertCacheLiterals, expert: Int
) {
    let gateUp = literals.gateUpWords(for: expert)
    let down = literals.downWords(for: expert)
    #expect(mapped.expertID == expert)
    #expect(bufferBytes(mapped.gateUp, count: mapped.gateUpLength)
        == QwenBF16TestFixture.littleEndianBytes(gateUp))
    #expect(bufferBytes(mapped.down, count: mapped.downLength)
        == QwenBF16TestFixture.littleEndianBytes(down))
}
