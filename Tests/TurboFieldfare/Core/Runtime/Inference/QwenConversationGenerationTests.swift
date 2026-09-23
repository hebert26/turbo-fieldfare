import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

/// Runtime-level coverage for the retained Qwen conversation adapter.  These
/// tests use the reviewed tiny text model and the same conversation state used
/// by production.  They do not call the URL loader or use authentic weights.
@Suite(.serialized) struct QwenConversationGenerationTests {
    @Test func retainedTextContinuationDoesNotDuplicateTheFirstPrompt() async throws {
        let fixture = try await makeFixture()
        let first = ModelChatMessage(role: .user, content: "first retained question")
        let second = ModelChatMessage(role: .user, content: "second retained question")

        let firstResult = try await fixture.session.generate(
            request(turn: .user(first), thinking: .disabled, maxNewTokens: 1),
            onEvent: { _ in })
        let afterFirst = await fixture.state.status()
        let firstPrompt = try fixture.codec.encodePrompt(
            messages: [first], tools: [], options: .init(enableThinking: false))
        let mappedFirstPrompt = try fixture.mapperRecorder.map(firstPrompt)

        let secondResult = try await fixture.session.generate(
            request(turn: .user(second), thinking: .disabled, maxNewTokens: 1),
            onEvent: { _ in })
        let afterSecond = await fixture.state.status()

        #expect(firstResult.promptTokens == firstPrompt.count)
        #expect(secondResult.promptTokens > 0)
        #expect(afterFirst.committed.retainedTokenIDs.starts(with: mappedFirstPrompt))
        #expect(afterSecond.committed.retainedTokenIDs.count > afterFirst.committed.retainedTokenIDs.count)
        // A second copy of the original prompt at the turn boundary would
        // show up as an immediate repeated prefix in the retained journal.
        let tail = Array(afterSecond.committed.retainedTokenIDs.dropFirst(
            mappedFirstPrompt.count + firstResult.newTokens))
        #expect(!tail.starts(with: mappedFirstPrompt))
        let expectedSuffix = try fixture.codec.encodeContinuation(
            messages: [second], boundary: .openAssistant,
            options: .init(enableThinking: false))
        let mappedExpectedSuffix = try fixture.mapperRecorder.map(expectedSuffix)
        #expect(tail.starts(with: mappedExpectedSuffix))
        #expect(fixture.mapperRecorder.values.contains {
            $0.composed == expectedSuffix && $0.mapped.count == expectedSuffix.count
        })
        #expect(afterSecond.activeTransaction == nil)
    }

    @Test func continuationBoundariesUseLiteralOfficialQwenSuffixes() async throws {
        let fixture = try await makeFixture()
        let next = ModelChatMessage(role: .user, content: "continue the retained answer")
        let options = ModelChatRenderOptions(enableThinking: false)

        // These strings are the pinned Qwen template output.  Keep the
        // expected side independent from encodeContinuation so a duplicated
        // or missing boundary token cannot make the test self-consistent.
        let endedText = "\n<|im_start|>user\ncontinue the retained answer"
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        let openText = "<|im_end|>\n<|im_start|>user\ncontinue the retained answer"
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        #expect(try fixture.codec.renderContinuation(
            messages: [next], boundary: .endedWithEndToken, options: options) == endedText)
        #expect(try fixture.codec.renderContinuation(
            messages: [next], boundary: .openAssistant, options: options) == openText)

        let eosFixture = try await makeFixture(
            fixtureGeneratedTokenIDs: [2], fixtureEndTokenID: 2)
        let ended = try await eosFixture.session.generate(
            request(
                turn: .user(.init(role: .user, content: "first answer")),
                thinking: .disabled, maxNewTokens: 1),
            onEvent: { _ in })
        #expect(ended.reason == .eos)
        #expect(ended.acceptedGeneratedTokenIDs == [2])
        let expectedEndedTokens = eosFixture.codec.tokenizer.encode(endedText)
        let afterEOS = try await eosFixture.session.generate(
            request(turn: .user(next), thinking: .disabled, maxNewTokens: 1),
            onEvent: { _ in })
        #expect(afterEOS.promptTokens == expectedEndedTokens.count)
        #expect(eosFixture.mapperRecorder.values.contains {
            $0.composed == expectedEndedTokens && $0.mapped.count == expectedEndedTokens.count
        })

        // A committed checkpoint establishes the open-assistant boundary.
        // Stopping before sampling leaves it open and lets the next turn
        // prove that this exact literal suffix becomes its prefill.
        let checkpointID = UUID()
        _ = try await fixture.session.rebuildCheckpoint(
            .init(
                checkpointID: checkpointID,
                messages: [.init(role: .user, content: "checkpoint answer")],
                thinking: .disabled, reason: .capacity, commit: true),
            onEvent: { _ in })
        let stopped = try await fixture.session.generate(
            request(turn: .checkpoint(checkpointID), thinking: .disabled, maxNewTokens: 1),
            shouldStop: { true }, onEvent: { _ in })
        #expect(stopped.reason == .cancelled)
        #expect(stopped.acceptedGeneratedTokenIDs.isEmpty)

        let expectedOpenTokens = fixture.codec.tokenizer.encode(openText)
        let nextTurn = try await fixture.session.generate(
            request(turn: .user(next), thinking: .disabled, maxNewTokens: 1),
            onEvent: { _ in })
        #expect(nextTurn.promptTokens == expectedOpenTokens.count)
        #expect(fixture.mapperRecorder.values.contains {
            $0.composed == expectedOpenTokens && $0.mapped.count == expectedOpenTokens.count
        })
    }

    @Test func systemPromptIsRenderedBeforeTheFirstUserTurnAndCannotChangeLater() async throws {
        let fixture = try await makeFixture()
        let system = "Answer briefly and use metric units."
        let user = ModelChatMessage(role: .user, content: "how far?")
        let expected = try fixture.codec.encodePrompt(
            messages: [
                .init(role: .system, content: system), user,
            ], tools: [], options: .init(enableThinking: false))
        let first = try await fixture.session.generate(
            QwenConversationGenerationRequest(
                turn: .user(user), systemPrompt: system, tools: [], imagesByID: [:],
                thinking: .disabled, visionResidency: .defaultPolicy,
                config: .init(maxNewTokens: 1, temperature: 0, topK: nil, topP: nil,
                              repetitionPenalty: 1, seed: 0)),
            onEvent: { _ in })
        #expect(first.promptTokens == expected.count)

        do {
            _ = try await fixture.session.generate(
                QwenConversationGenerationRequest(
                    turn: .user(.init(role: .user, content: "follow up")),
                    systemPrompt: "Use imperial units instead.", tools: [],
                    imagesByID: [:], thinking: .disabled,
                    visionResidency: .defaultPolicy,
                    config: .init(maxNewTokens: 1, temperature: 0, topK: nil, topP: nil,
                                  repetitionPenalty: 1, seed: 0)),
                onEvent: { _ in })
            Issue.record("changed system prompt was accepted on a retained conversation")
        } catch {
            #expect((error as? QwenConversationGenerationError)
                == .systemPromptChanged)
        }
    }

    @Test func thinkingEnabledAndDisabledUseTheirDistinctChatPromptModes() async throws {
        let disabledFixture = try await makeFixture()
        let enabledFixture = try await makeFixture()
        let message = ModelChatMessage(role: .user, content: "show the answer")

        let disabled = try await disabledFixture.session.rebuildCheckpoint(
            .init(
                checkpointID: UUID(),
                messages: [message], tools: [], imagesByID: [:], thinking: .disabled,
                visionResidency: .defaultPolicy, reason: .capacity, commit: false),
            onEvent: { _ in })
        let enabled = try await enabledFixture.session.rebuildCheckpoint(
            .init(
                checkpointID: UUID(),
                messages: [message], tools: [], imagesByID: [:], thinking: .enabled,
                visionResidency: .defaultPolicy, reason: .capacity, commit: false),
            onEvent: { _ in })

        let disabledPrompt = try disabledFixture.codec.encodePrompt(
            messages: [message], tools: [],
            options: .init(enableThinking: false, preserveThinking: true))
        let enabledPrompt = try enabledFixture.codec.encodePrompt(
            messages: [message], tools: [],
            options: .init(enableThinking: true, preserveThinking: true))
        #expect(disabled.committed == false)
        #expect(enabled.committed == false)
        #expect(disabled.promptTokens == disabledPrompt.count)
        #expect(enabled.promptTokens == enabledPrompt.count)
        // The official template's thinking-on generation prompt is the open
        // `<think>` channel. Thinking-off adds the explicit empty thought
        // close, so its token vector is longer for this same user turn.
        #expect(enabled.promptTokens < disabled.promptTokens)
        #expect(disabled.metrics.retainedTokenIDs.isEmpty)
        #expect(enabled.metrics.retainedTokenIDs.isEmpty)
        #expect((await disabledFixture.state.status()).committed.retainedTokenIDs.isEmpty)
        #expect((await enabledFixture.state.status()).committed.retainedTokenIDs.isEmpty)
    }

    @Test func orderedToolSchemaAssistantCallAndResultsSurviveCheckpointRebuild() async throws {
        let fixture = try await makeFixture()
        let tool = lookupTool()
        let checkpointID = UUID()
        let call = ModelChatToolCall(
            id: "call-1", name: "lookup",
            arguments: .object([.init("query", .string("snow"))]))
        let messages = [
            ModelChatMessage(role: .user, content: "find the weather"),
            ModelChatMessage(role: .assistant, content: "", toolCalls: [call]),
            ModelChatMessage(
                role: .tool, content: "{\"temperature\": 2}",
                toolCallID: "call-1", name: "lookup"),
            ModelChatMessage(role: .user, content: "summarize it"),
        ]
        let expected = try fixture.codec.encodePrompt(
            messages: messages, tools: [tool],
            options: .init(enableThinking: false, preserveThinking: true))
        let expectedMapped = try fixture.mapperRecorder.map(expected)

        let preview = try await fixture.session.rebuildCheckpoint(
            .init(
                checkpointID: UUID(),
                messages: messages, tools: [tool], imagesByID: [:], thinking: .disabled,
                visionResidency: .defaultPolicy, reason: .capacity, commit: false),
            onEvent: { _ in })
        #expect(preview.committed == false)
        #expect(preview.promptTokens == expected.count)
        #expect(preview.retainedImageCount == 0)
        #expect(preview.retainedImageRows == 0)
        #expect(preview.retainedFeatureBytes == 0)
        #expect(preview.metrics.retainedTokenIDs.isEmpty)
        #expect((await fixture.state.status()).committed.retainedTokenIDs.isEmpty)

        let committed = try await fixture.session.rebuildCheckpoint(
            .init(
                checkpointID: checkpointID,
                messages: messages, tools: [tool], imagesByID: [:], thinking: .disabled,
                visionResidency: .defaultPolicy, reason: .capacity, commit: true),
            onEvent: { _ in })
        #expect(committed.committed)
        #expect(committed.promptTokens == expected.count)
        #expect(committed.retainedImageCount == 0)
        #expect(committed.retainedImageRows == 0)
        #expect(committed.retainedFeatureBytes == 0)
        #expect(committed.metrics.retainedTokenIDs == expectedMapped)
        #expect((await fixture.state.status()).committed.retainedTokenIDs == expectedMapped)
        #expect(messages[1].toolCalls.first?.id == "call-1")
        #expect(messages[2].toolCallID == "call-1")

        _ = try await fixture.session.generate(
            request(
                turn: .checkpoint(checkpointID), thinking: .disabled,
                maxNewTokens: 1, tools: [tool]),
            onEvent: { _ in })

        let nextToolResult = ModelChatMessage(
            role: .tool, content: "{\"humidity\": 40}",
            toolCallID: "call-1", name: "lookup")
        let continuation = try fixture.codec.encodeContinuation(
            messages: [nextToolResult], boundary: .openAssistant,
            options: .init(enableThinking: false))
        let toolTurn = try await fixture.session.generate(
            request(
                turn: .toolResults([nextToolResult]), thinking: .disabled,
                maxNewTokens: 1, tools: [tool]),
            onEvent: { _ in })
        #expect(toolTurn.promptTokens == continuation.count)
        #expect((await fixture.state.status()).activeTransaction == nil)

        let changedTool = ModelChatToolDefinition(function: .init(
            name: "lookup", description: "changed schema",
            parameters: tool.function.parameters))
        do {
            _ = try await fixture.session.generate(
                request(
                    turn: .toolResults([nextToolResult]), thinking: .disabled,
                    maxNewTokens: 1, tools: [changedTool]),
                onEvent: { _ in })
            Issue.record("a changed retained tool schema was accepted")
        } catch let error as QwenConversationGenerationError {
            #expect(error == .toolsChanged)
        }
    }

    @Test func terminalToolPublicationUsesTheValidatedEOSBoundary() throws {
        let tool = lookupTool()
        let frame = "<tool_call>\n<function=lookup>\n<parameter=query>\n"
            + "snow\n</parameter>\n</function>\n</tool_call>"

        var atMaxTokens = QwenStructuredAssistantDecoder(
            tools: [tool], startsInThoughtChannel: false,
            idGenerator: { "host-call" })
        _ = try atMaxTokens.consume(frame)
        var nonTerminal: [StructuredAssistantEvent] = []
        try finalizeQwenStructuredTurn(
            decoder: &atMaxTokens, tokenizerTail: "", termination: .maxTokens) {
                nonTerminal.append($0)
            }
        #expect(nonTerminal.isEmpty)

        var atEOS = QwenStructuredAssistantDecoder(
            tools: [tool], startsInThoughtChannel: false,
            idGenerator: { "host-call" })
        _ = try atEOS.consume(frame)
        var terminal: [StructuredAssistantEvent] = []
        try finalizeQwenStructuredTurn(
            decoder: &atEOS, tokenizerTail: "", termination: .modelEOS) {
                terminal.append($0)
            }
        #expect(terminal == [.toolCall(.init(
            id: "host-call", name: "lookup",
            arguments: .object(["query": .string("snow")]),
            argumentsJSON: "{\"query\":\"snow\"}"))])
    }

    @Test func committedCheckpointResumesOnceAndFailureLeavesItsIDRetryable() async throws {
        let fixture = try await makeFixture()
        let checkpointID = UUID()
        let message = ModelChatMessage(role: .user, content: "resume this checkpoint")
        let checkpoint = try await fixture.session.rebuildCheckpoint(
            .init(
                checkpointID: checkpointID, messages: [message], tools: [],
                imagesByID: [:], thinking: .disabled, visionResidency: .defaultPolicy,
                reason: .sustainedSlowDecode, commit: true),
            onEvent: { _ in })
        #expect(checkpoint.committed)

        do {
            _ = try await fixture.session.generate(
                request(
                    turn: .user(.init(role: .user, content: "must wait")),
                    thinking: .disabled, maxNewTokens: 1),
                onEvent: { _ in })
            Issue.record("a user turn was accepted while a checkpoint was pending")
        } catch let error as QwenConversationGenerationError {
            guard case .invalidTurn = error else {
                Issue.record("unexpected pending-checkpoint error: \(error)")
                return
            }
        }

        let mismatches = [
            QwenConversationGenerationRequest(
                turn: .checkpoint(UUID()), tools: [], imagesByID: [:],
                thinking: .disabled, visionResidency: .defaultPolicy,
                config: .init(maxNewTokens: 1, temperature: 0, topK: nil, topP: nil,
                              repetitionPenalty: 1, seed: 0)),
            QwenConversationGenerationRequest(
                turn: .checkpoint(checkpointID), tools: [], imagesByID: [:],
                thinking: .enabled, visionResidency: .defaultPolicy,
                config: .init(maxNewTokens: 1, temperature: 0, topK: nil, topP: nil,
                              repetitionPenalty: 1, seed: 0)),
            QwenConversationGenerationRequest(
                turn: .checkpoint(checkpointID), tools: [],
                imagesByID: ["unexpected": URL(fileURLWithPath: "/fixture/unexpected.png")],
                thinking: .disabled, visionResidency: .defaultPolicy,
                config: .init(maxNewTokens: 1, temperature: 0, topK: nil, topP: nil,
                              repetitionPenalty: 1, seed: 0)),
        ]
        for mismatch in mismatches {
            do {
                _ = try await fixture.session.generate(mismatch, onEvent: { _ in })
                Issue.record("a mismatched checkpoint request was accepted")
            } catch let error as QwenConversationGenerationError {
                guard case .invalidTurn = error else {
                    Issue.record("unexpected checkpoint mismatch error: \(error)")
                    return
                }
            }
        }

        let invalidConfig = QwenConversationGenerationRequest(
            turn: .checkpoint(checkpointID), tools: [], imagesByID: [:],
            thinking: .disabled, visionResidency: .defaultPolicy,
            config: .init(maxNewTokens: 0, temperature: 0, topK: nil, topP: nil,
                          repetitionPenalty: 1, seed: 0))
        await #expect(throws: GeneratorError.self) {
            try await fixture.session.generate(invalidConfig, onEvent: { _ in })
        }

        let resumed = try await fixture.session.generate(
            request(
                turn: .checkpoint(checkpointID), thinking: .disabled,
                maxNewTokens: 1),
            onEvent: { _ in })
        #expect(resumed.promptTokens == 0)
        #expect((await fixture.state.status()).activeTransaction == nil)

        do {
            _ = try await fixture.session.generate(
                request(
                    turn: .checkpoint(checkpointID), thinking: .disabled,
                    maxNewTokens: 1),
                onEvent: { _ in })
            Issue.record("a committed checkpoint was resumed twice")
        } catch let error as QwenConversationGenerationError {
            guard case .invalidTurn = error else {
                Issue.record("unexpected reused-checkpoint error: \(error)")
                return
            }
        }
    }

    @Test func hardCancellationRollsBackTheActiveTurnAndAllowsRetry() async throws {
        let fixture = try await makeFixture()
        let before = await fixture.state.status()
        let observedPrefill = LockedFlag()
        let operation = Task {
            try await fixture.session.generate(
                request(
                    turn: .user(.init(role: .user, content: String(repeating: "cancel ", count: 24))),
                    thinking: .disabled, maxNewTokens: 16),
                shouldStop: { false },
                onEvent: { event in
                    if case .prefill = event { observedPrefill.set() }
                })
        }

        for _ in 0..<100_000 where !observedPrefill.value {
            await Task.yield()
        }
        #expect(observedPrefill.value)
        operation.cancel()
        let outcome = await operation.result
        guard case .failure(let error) = outcome else {
            Issue.record("cancelled generation unexpectedly committed")
            return
        }
        #expect(error is CancellationError)
        #expect(await fixture.state.status() == before)
        #expect((await fixture.state.status()).activeTransaction == nil)

        let retry = try await fixture.session.generate(
            request(
                turn: .user(.init(role: .user, content: "retry after cancel")),
                thinking: .disabled, maxNewTokens: 1),
            onEvent: { _ in })
        #expect(retry.newTokens >= 0)
        #expect((await fixture.state.status()).activeTransaction == nil)
    }

    @Test func checkpointRebuildCancellationRollsBackAndCanRetry() async throws {
        let fixture = try await makeFixture()
        _ = try await fixture.session.generate(
            request(
                turn: .user(.init(role: .user, content: "existing retained state")),
                thinking: .disabled, maxNewTokens: 1),
            onEvent: { _ in })
        let before = await fixture.state.status()
        let checkpointID = UUID()
        let messages = [ModelChatMessage(
            role: .user,
            content: String(repeating: "checkpoint rebuild ", count: 64))]
        let observedProgress = LockedFlag()
        let operation = Task {
            try await fixture.session.rebuildCheckpoint(
                .init(
                    checkpointID: checkpointID, messages: messages, tools: [],
                    imagesByID: [:], thinking: .disabled,
                    visionResidency: .defaultPolicy, reason: .capacity, commit: true),
                onEvent: { event in
                    if case .progress(let done, _) = event, done > 0 {
                        observedProgress.set()
                    }
                })
        }

        for _ in 0..<100_000 where !observedProgress.value {
            await Task.yield()
        }
        #expect(observedProgress.value)
        operation.cancel()
        let outcome = await operation.result
        guard case .failure(let error) = outcome else {
            Issue.record("cancelled checkpoint rebuild unexpectedly committed")
            return
        }
        #expect(error is CancellationError)
        #expect(await fixture.state.status() == before)
        #expect((await fixture.state.status()).activeTransaction == nil)

        let retry = try await fixture.session.rebuildCheckpoint(
            .init(
                checkpointID: checkpointID, messages: messages, tools: [],
                imagesByID: [:], thinking: .disabled,
                visionResidency: .defaultPolicy, reason: .capacity, commit: true),
            onEvent: { _ in })
        #expect(retry.committed)
        #expect(retry.promptTokens > 0)
        #expect((await fixture.state.status()).activeTransaction == nil)
    }

    @Test func softStopCommitsOnlyTheAcceptedTokenBoundary() async throws {
        let fixture = try await makeFixture()
        let stop = LockedFlag()
        let events = EventRecorder()
        let result = try await fixture.session.generate(
            request(
                turn: .user(.init(role: .user, content: "stop after the first answer token")),
                thinking: .disabled, maxNewTokens: 16),
            shouldStop: { stop.value },
            onEvent: { event in
                events.append(event)
                switch event {
                case .text(let text) where !text.isEmpty:
                    stop.set()
                case .structuredProgress(let progress)
                    where progress.visibleResponseTokens > 0 || progress.thinkingTokens > 0:
                    stop.set()
                default:
                    break
                }
            })

        #expect(result.reason == .cancelled)
        #expect(!result.acceptedGeneratedTokenIDs.isEmpty)
        #expect(result.metrics.pendingTokenCount == 1)
        #expect(events.values.contains {
            if case .prefill = $0 { return true }
            return false
        })
        #expect((await fixture.state.status()).committed == result.metrics)
    }

    @Test func imageOrderAndBindingAreRetainedThroughGenerationAndCheckpointRebuild() async throws {
        let planner = try FixtureImagePlanner()
        let fixture = try await makeFixture(imagePlanner: planner.make)
        let messages = [ModelChatMessage(
            role: .user,
            content: .parts([
                .text("compare"), .image(.init(id: "first")),
                .text("with"), .image(.init(id: "second")),
            ]))]
        let request = QwenConversationGenerationRequest(
            turn: .user(messages[0]), tools: [],
            imagesByID: [
                "first": URL(fileURLWithPath: "/fixture/first.png"),
                "second": URL(fileURLWithPath: "/fixture/second.png"),
            ], thinking: .disabled, visionResidency: .onDemand,
            config: .init(maxNewTokens: 1, temperature: 0, topK: nil, topP: nil,
                          repetitionPenalty: 1, seed: 0))
        let preflight = try await fixture.session.preflightCheckpointImages(
            orderedImageIDs: ["first", "second"], imagesByID: request.imagesByID,
            visionResidency: .onDemand)
        #expect(preflight.retainedImageCount == 2)
        #expect(preflight.retainedImageRows == 2)
        #expect(preflight.retainedFeatureBytes == 2 * (
            32 * MemoryLayout<Float>.stride
                + 3 * MemoryLayout<Int32>.stride))
        let beforeGeneration = try await fixture.state.diagnosticSnapshot()
        #expect(beforeGeneration.lineage.visibleRows == 0)
        let preview = try await fixture.session.rebuildCheckpoint(
            .init(
                checkpointID: UUID(), messages: messages, imagesByID: request.imagesByID,
                thinking: .disabled, visionResidency: .onDemand,
                reason: .capacity, commit: false),
            onEvent: { _ in })
        #expect(!preview.committed)
        #expect(preview.retainedImageCount == 2)
        #expect(preview.retainedImageRows == 2)
        #expect(preview.retainedFeatureBytes == 2 * (
            32 * MemoryLayout<Float>.stride
                + 3 * MemoryLayout<Int32>.stride))
        let afterPreview = try await fixture.state.diagnosticSnapshot()
        #expect(afterPreview.lineage.visibleRows == 0)
        let generated = try await fixture.session.generate(request, onEvent: { _ in })
        #expect(generated.metrics.pendingTokenCount <= 1)
        let imageCalls = fixture.mapperRecorder.values.filter { call in
            call.composed.contains(Int32(QwenTestArchitecture.qwen36.visionStartTokenID))
                || call.composed.contains(Int32(QwenTestArchitecture.qwen36.imageTokenID))
                || call.composed.contains(Int32(QwenTestArchitecture.qwen36.visionEndTokenID))
        }
        #expect(!imageCalls.isEmpty)
        if let imageCall = imageCalls.last {
            let sourceStart = Int32(QwenTestArchitecture.qwen36.visionStartTokenID)
            let sourceImage = Int32(QwenTestArchitecture.qwen36.imageTokenID)
            let sourceEnd = Int32(QwenTestArchitecture.qwen36.visionEndTokenID)
            let targetStart = Int32(FixtureImagePlanner.tinyImageArchitecture.visionStartTokenID)
            let targetImage = Int32(FixtureImagePlanner.tinyImageArchitecture.imageTokenID)
            let targetEnd = Int32(FixtureImagePlanner.tinyImageArchitecture.visionEndTokenID)
            for (index, token) in imageCall.composed.enumerated() {
                switch token {
                case sourceStart: #expect(imageCall.mapped[index] == targetStart)
                case sourceImage: #expect(imageCall.mapped[index] == targetImage)
                case sourceEnd: #expect(imageCall.mapped[index] == targetEnd)
                default: break
                }
            }
        }
        let generatedSnapshot = try await fixture.state.diagnosticSnapshot()
        #expect(generatedSnapshot.lineage.visibleRows == 2)
        #expect(generatedSnapshot.lineage.occurrences.map(\.ownerAllocationID).count == 2)
        #expect(generatedSnapshot.lineage.occurrences.map(\.exactPositions).count == 2)

        let rebuilt = try await fixture.session.rebuildCheckpoint(
            .init(
                checkpointID: UUID(),
                messages: messages, tools: [], imagesByID: request.imagesByID,
                thinking: .disabled, visionResidency: .onDemand,
                reason: .capacity, commit: true),
            onEvent: { _ in })
        #expect(rebuilt.committed)
        #expect(rebuilt.retainedImageCount == 2)
        #expect(rebuilt.retainedImageRows == 2)
        #expect(rebuilt.retainedFeatureBytes == 2 * (
            32 * MemoryLayout<Float>.stride
                + 3 * MemoryLayout<Int32>.stride))
        let rebuiltSnapshot = try await fixture.state.diagnosticSnapshot()
        #expect(rebuiltSnapshot.lineage.visibleRows == 2)
        #expect(rebuiltSnapshot.lineage.occurrences.count == 2)
        #expect(rebuiltSnapshot.lineage.occurrences.map(\.exactPositions)
            == generatedSnapshot.lineage.occurrences.map(\.exactPositions))
        #expect(rebuiltSnapshot.lineage.occurrences.map(\.ownerAllocationID)
            != generatedSnapshot.lineage.occurrences.map(\.ownerAllocationID))
    }

    @Test func identityMismatchIsRejectedBeforeMutatingRetainedState() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(model: model, context: context, maxContext: 128)
        let codec = try QwenP18FixtureSupport.codec()
        let before = await state.status()
        let wrongIdentity = try makeIdentity(textManifestSHA256: String(repeating: "f", count: 64))

        #expect(throws: ModelFamilyGenerationError.modelIdentityChanged) {
            try QwenConversationGenerationSession(
                model: model, state: state, codec: codec,
                verifiedIdentity: wrongIdentity,
                modelDirectoryURL: fixtureSourceDirectory,
                context: context, maxContext: 128)
        }
        #expect(await state.status() == before)
    }

    @Test func oneInjectedModelAndStateAreReusedAcrossTurns() async throws {
        let fixture = try await makeFixture()
        let modelIdentityBefore = (try await fixture.state.diagnosticSnapshot()).runnerState.architectureIdentity
        let first = try await fixture.session.generate(
            request(
                turn: .user(.init(role: .user, content: "reuse one loaded model")),
                thinking: .disabled, maxNewTokens: 1),
            onEvent: { _ in })
        let second = try await fixture.session.generate(
            request(
                turn: .user(.init(role: .user, content: "reuse it again")),
                thinking: .disabled, maxNewTokens: 1),
            onEvent: { _ in })
        let after = try await fixture.state.diagnosticSnapshot()
        #expect(first.metrics.logicalStateBytes > 0)
        #expect(second.metrics.logicalStateBytes > 0)
        #expect(after.runnerState.architectureIdentity == modelIdentityBefore)
        #expect(after.retainedTokenIDs.count > first.metrics.retainedTokenIDs.count)
        #expect(after.runnerState.sequenceLength > 0)
    }

    private func makeFixture(
        // The official tool preamble is substantially longer than the tiny
        // fixture vocabulary. Leave room for it and one bounded decode turn.
        maxContext: Int = 2_048,
        imagePlanner: QwenConversationFixtureImagePlanner? = nil,
        fixtureGeneratedTokenIDs: [Int32]? = Self.defaultFixtureGeneratedTokenIDs,
        fixtureEndTokenID: Int32? = nil
    ) async throws -> Fixture {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: maxContext)
        let codec = try QwenP18FixtureSupport.codec()
        let mapperRecorder = TokenMapperRecorder()
        let session = try QwenConversationGenerationSession(
            fixtureModel: model, state: state, codec: codec, context: context,
            maxContext: maxContext, imagePlanner: imagePlanner,
            fixtureTokenMapper: mapperRecorder.map,
            fixtureGeneratedTokenIDs: fixtureGeneratedTokenIDs,
            fixtureEndTokenID: fixtureEndTokenID)
        return Fixture(
            session: session, state: state, codec: codec, context: context,
            mapperRecorder: mapperRecorder)
    }

    private static let defaultFixtureGeneratedTokenIDs: [Int32] =
        Array(repeating: [Int32(1), 12, 6], count: 16).flatMap { $0 }

    private func request(
        turn: QwenConversationTurn,
        thinking: ModelFamilyThinkingMode,
        maxNewTokens: Int,
        tools: [ModelChatToolDefinition] = []
    ) -> QwenConversationGenerationRequest {
        QwenConversationGenerationRequest(
            turn: turn, systemPrompt: nil, tools: tools, imagesByID: [:], thinking: thinking,
            visionResidency: .defaultPolicy,
            config: .init(maxNewTokens: maxNewTokens, temperature: 0, topK: nil, topP: nil,
                          repetitionPenalty: 1, seed: 0))
    }

    private func lookupTool() -> ModelChatToolDefinition {
        let schema = ModelChatJSONValue.object([
            .init("type", .string("object")),
            .init("properties", .object([
                .init("query", .object([.init("type", .string("string"))])),
            ])),
            .init("required", .array([.string("query")])),
            .init("additionalProperties", .bool(false)),
        ])
        return ModelChatToolDefinition(function: .init(
            name: "lookup", description: "decoder test tool", parameters: schema))
    }

    private struct Fixture {
        let session: QwenConversationGenerationSession
        let state: QwenConversationState
        let codec: QwenChatCodec
        let context: MetalContext
        let mapperRecorder: TokenMapperRecorder
    }

    private static var fixtureSourceDirectory: URL {
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(
                "scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0",
                isDirectory: true)
    }

    private var fixtureSourceDirectory: URL { Self.fixtureSourceDirectory }

    fileprivate struct FixtureImagePlanner {
        let context: MetalContext

        init() throws { context = try MetalContext() }

        var make: QwenConversationFixtureImagePlanner {
            let context = context
            return { orderedIDs, _ in
                let requestedAllocationBytes = orderedIDs.count * (
                    32 * MemoryLayout<Float>.stride
                        + 3 * MemoryLayout<Int32>.stride)
                return QwenConversationFixtureImagePlan(
                    architecture: Self.tinyImageArchitecture,
                    visionConfig: QwenVisionConfig(
                        outputHiddenSize: 32, allowsFixtureGeometry: true),
                    mergedRows: Array(repeating: 1, count: orderedIDs.count),
                    requestedAllocationBytes: requestedAllocationBytes,
                    prepare: {
                        try orderedIDs.enumerated().map { index, _ in
                            try Self.qwenFixtureFeatures(
                                context: context, marker: Float(index + 1))
                        }
                    })
            }
        }

        static let tinyImageArchitecture: QwenArchConfig = {
            let wire = GTurboQwenArchitectureV2(
                hiddenSize: 32, numLayers: 4,
                layerTypes: [.linearAttention, .linearAttention,
                              .linearAttention, .fullAttention],
                numAttentionHeads: 2, numKeyValueHeads: 1, headDimension: 16,
                attentionOutputGate: true, linearConvolutionKernel: 4,
                linearKeyHeads: 2, linearKeyHeadDimension: 4,
                linearValueHeads: 2, linearValueHeadDimension: 4,
                recurrentStateType: .fp32, partialRotaryFactor: 0.75,
                ropeTheta: 10_000_000, mropeInterleaved: true,
                mropeSections: [2, 1, 3], numberOfExperts: 10,
                expertsPerToken: 8, routedExpertIntermediateSize: 7,
                sharedExpertIntermediateSize: 6, vocabularySize: 19,
                tiedWordEmbeddings: false, hiddenActivation: "silu",
                bosTokenID: 1, eosTokenID: 2, imageTokenID: 3,
                videoTokenID: 4, visionStartTokenID: 5,
                visionEndTokenID: 6)
            return QwenArchConfig(wire: wire)
        }()

        static func qwenFixtureFeatures(
            context: MetalContext, marker: Float
        ) throws -> QwenVisionFeatures {
            let grid = try QwenVisionGrid(temporal: 1, height: 2, width: 2)
            let position = try QwenMRoPEPosition(temporal: 0, height: 0, width: 0)
            let profile = GTurboQwenVisionProcessorProfileV2(
                processorClass: "Qwen3VLProcessor",
                imageProcessorType: "Qwen2VLImageProcessorFast",
                patchSize: 16, temporalPatchSize: 2, spatialMergeSize: 2)
            return try QwenVisionFeatures(
                device: context.device,
                features: [Float](repeating: marker, count: 32),
                positions: [position],
                imageDigest: String(repeating: marker == 1 ? "a" : "b", count: 64),
                processorDigest: String(repeating: "c", count: 64),
                profile: profile, grid: grid, hiddenSize: 32)
        }
    }

    private func makeIdentity(textManifestSHA256: String) throws -> LoadedRuntimeIdentity {
        let architecture = GTurboQwenArchitectureV2(
            hiddenSize: 2_048, numLayers: 40,
            layerTypes: (0..<40).map { ($0 + 1).isMultiple(of: 4) ? .fullAttention : .linearAttention },
            numAttentionHeads: 16, numKeyValueHeads: 2, headDimension: 256,
            attentionOutputGate: true, linearConvolutionKernel: 4,
            linearKeyHeads: 16, linearKeyHeadDimension: 128,
            linearValueHeads: 32, linearValueHeadDimension: 128,
            recurrentStateType: .fp32, partialRotaryFactor: 0.25,
            ropeTheta: 10_000_000, mropeInterleaved: true,
            mropeSections: [11, 11, 10], numberOfExperts: 256,
            expertsPerToken: 8, routedExpertIntermediateSize: 512,
            sharedExpertIntermediateSize: 512, vocabularySize: 248_320,
            tiedWordEmbeddings: false, hiddenActivation: "silu",
            bosTokenID: 248_044, eosTokenID: 248_044, imageTokenID: 248_056,
            videoTokenID: 248_057, visionStartTokenID: 248_053,
            visionEndTokenID: 248_054)
        let digest = String(repeating: "a", count: 64)
        let groups = GTurboQuantizationCategoryV2.allCases.map { category in
            category == .recurrentState
                ? GTurboQuantizationGroupV2(category: category, storage: .fp32)
                : GTurboQuantizationGroupV2(
                    category: category, storage: .affineInt4, groupSize: 64,
                    scaleType: "bf16", biasType: "bf16")
        }
        let manifest = GTurboManifestV2(
            family: .qwen3_6,
            requiredFeatures: [.familyDispatch, .verifiedIdentity,
                               .qwenHybridAttention, .qwenMTPExcluded],
            modelID: GTurboFormatV2.qwenRepository,
            architecture: .qwen3_6(architecture),
            provenance: .init(
                sourceRepository: GTurboFormatV2.qwenRepository,
                sourceRevision: GTurboFormatV2.qwenRevision,
                sourceIndexSHA256: GTurboFormatV2.qwenSourceIndexSHA256,
                sidecarSHA256: GTurboFormatV2.qwenSidecarSHA256,
                quantizationPolicySHA256: digest),
            quantization: groups, ignoredTensors: [], files: [:],
            tensorRegions: [], expertsPerLayer: 256, numLayers: 40,
            expertStride: 16_384)
        let descriptor = try InstalledModelDescriptor.validated(
            textManifest: manifest, textManifestSHA256: textManifestSHA256)
        return LoadedRuntimeIdentity(descriptor: descriptor)
    }
}

private final class LockedFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var storage = false

    var value: Bool { lock.withLock { storage } }

    func set() { lock.withLock { storage = true } }
}

private final class EventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [QwenConversationGenerationEvent] = []

    func append(_ event: QwenConversationGenerationEvent) {
        lock.withLock { storage.append(event) }
    }

    var values: [QwenConversationGenerationEvent] { lock.withLock { storage } }
}

fileprivate final class TokenMapperRecorder: @unchecked Sendable {
    struct Call: Sendable {
        let composed: [Int32]
        let mapped: [Int32]
    }

    private let lock = NSLock()
    private var storage: [Call] = []

    func map(_ composed: [Int32]) throws -> [Int32] {
        let sourceStart = Int32(QwenTestArchitecture.qwen36.visionStartTokenID)
        let sourceImage = Int32(QwenTestArchitecture.qwen36.imageTokenID)
        let sourceEnd = Int32(QwenTestArchitecture.qwen36.visionEndTokenID)
        let targetStart = Int32(
            QwenConversationGenerationTests.FixtureImagePlanner
                .tinyImageArchitecture.visionStartTokenID)
        let targetImage = Int32(
            QwenConversationGenerationTests.FixtureImagePlanner
                .tinyImageArchitecture.imageTokenID)
        let targetEnd = Int32(
            QwenConversationGenerationTests.FixtureImagePlanner
                .tinyImageArchitecture.visionEndTokenID)
        let safeTextTokens: [Int32] = [0, 1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        let mapped = composed.map { token in
            switch token {
            case sourceStart: targetStart
            case sourceImage: targetImage
            case sourceEnd: targetEnd
            default: safeTextTokens[Int(token) % safeTextTokens.count]
            }
        }
        lock.withLock { storage.append(Call(composed: composed, mapped: mapped)) }
        return mapped
    }

    var values: [Call] { lock.withLock { storage } }
}
