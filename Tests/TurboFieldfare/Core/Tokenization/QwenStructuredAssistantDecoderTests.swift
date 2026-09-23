import Foundation
import Synchronization
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

@Suite("Qwen structured assistant decoder")
struct QwenStructuredAssistantDecoderTests {
    private func schema(
        _ properties: [(String, ModelChatJSONValue)],
        required: [String] = []
    ) -> ModelChatJSONValue {
        .object([
            .init("type", .string("object")),
            .init("properties", .object(properties.map { .init($0.0, $0.1) })),
            .init("required", .array(required.map { .string($0) })),
            .init("additionalProperties", .bool(false)),
        ])
    }

    private func stringSchema() -> ModelChatJSONValue {
        .object([.init("type", .string("string"))])
    }

    private func tool(
        _ name: String = "lookup",
        properties: [(String, ModelChatJSONValue)]? = nil,
        required: [String] = ["query"]
    ) -> ModelChatToolDefinition {
        let effectiveProperties = properties ?? [("query", stringSchema())]
        return ModelChatToolDefinition(function: ModelChatFunctionDefinition(
            name: name,
            description: "decoder test tool",
            parameters: schema(effectiveProperties, required: required)))
    }

    private func frame(
        function: String = "lookup",
        parameters: [(String, String)] = [("query", "snow")]
    ) -> String {
        var text = "<tool_call>\n<function=\(function)>\n"
        for (name, body) in parameters {
            text += "<parameter=\(name)>\n\(body)\n</parameter>\n"
        }
        return text + "</function>\n</tool_call>"
    }

    private func decoder(
        startsInThoughtChannel: Bool = false,
        tools: [ModelChatToolDefinition] = [],
        idGenerator: @escaping @Sendable () -> String = { "host-id" }
    ) -> QwenStructuredAssistantDecoder {
        QwenStructuredAssistantDecoder(
            tools: tools,
            startsInThoughtChannel: startsInThoughtChannel,
            idGenerator: idGenerator)
    }

    private final class InvocationCounter: Sendable {
        private let storage = Mutex(0)

        var value: Int { storage.withLock { $0 } }

        func increment() -> Int {
            storage.withLock {
                $0 += 1
                return $0
            }
        }
    }

    private func countingGenerator(_ invocations: InvocationCounter) -> @Sendable () -> String {
        {
            "host-\(invocations.increment())"
        }
    }

    @Test func releasedCallRoundTripsThroughQwenAssistantAndToolResultFraming() throws {
        let value = "雪<&>'\nnext"
        let optionsSchema = ModelChatJSONValue.object([
            .init("type", .string("object")),
            .init("properties", .object([
                .init("limit", .object([.init("type", .string("integer"))])),
            ])),
            .init("required", .array([.string("limit")])),
            .init("additionalProperties", .bool(false)),
        ])
        let roundTripTool = tool(
            properties: [("query", stringSchema()), ("options", optionsSchema)],
            required: ["query", "options"])
        var decoder = decoder(tools: [roundTripTool])
        _ = try decoder.consume(frame(parameters: [
            ("query", value), ("options", "{\"limit\":3}"),
        ]))
        let events = try decoder.finish()
        guard case let .toolCall(parsed) = events.single else {
            Issue.record("expected one released Qwen tool call")
            return
        }
        #expect(parsed.id == "host-id")
        #expect(parsed.arguments == .object([
            "query": .string(value),
            "options": .object(["limit": .integer(3)]),
        ]))

        let codec = try localQwenCodec()
        let call = ModelChatToolCall(
            id: parsed.id,
            name: parsed.name,
            arguments: try modelChatValue(from: parsed.arguments))
        let toolResult = ModelChatMessage(
            role: .tool,
            content: "result",
            toolCallID: parsed.id,
            name: parsed.name)
        let rendered = try codec.renderPrompt(
            messages: [
                ModelChatMessage(role: .user, content: "look up snow"),
                ModelChatMessage(role: .assistant, content: "", toolCalls: [call]),
                toolResult,
            ],
            tools: [roundTripTool],
            options: .init(addGenerationPrompt: false, enableThinking: false))

        #expect(rendered.contains("<parameter=query>\n\(value)\n</parameter>"))
        #expect(rendered.contains("<parameter=options>\n{\"limit\": 3}\n</parameter>"))
        #expect(rendered.contains("</function>\n</tool_call>"))
        #expect(rendered.contains("<tool_response>\nresult\n</tool_response>"))

        guard let start = rendered.range(of: "<tool_call>", options: .backwards),
              let relativeEnd = rendered[start.upperBound...].range(of: "</tool_call>") else {
            Issue.record("expected serialized tool-call frame")
            return
        }
        let frameEnd = relativeEnd.upperBound
        let reparsed = try QwenToolCallParser().parse(
            String(rendered[start.lowerBound..<frameEnd]),
            tools: [roundTripTool],
            id: "reparsed")
        #expect(reparsed.arguments == parsed.arguments)
        // Qwen framing deliberately omits the host ID; the host association
        // remains on both ModelChatToolCall.id and the tool-result message.
        #expect(!rendered.contains(parsed.id))
        #expect(call.id == parsed.id)
        #expect(toolResult.toolCallID == parsed.id)
    }

    @Test func completeFrameIsBufferedUntilExplicitFinishAndGetsHostID() throws {
        var decoder = decoder(tools: [tool()])
        var streamed: [StructuredAssistantEvent] = []
        for character in frame() {
            streamed += try decoder.consume(String(character))
        }

        #expect(streamed.isEmpty)
        #expect(!decoder.hasToolCalls)
        let released = try decoder.finish()
        #expect(released == [.toolCall(ParsedToolCall(
            id: "host-id",
            name: "lookup",
            arguments: .object(["query": .string("snow")]),
            argumentsJSON: "{\"query\":\"snow\"}"))])
        #expect(decoder.hasToolCalls)
        #expect(try decoder.finish().isEmpty)
    }

    @Test func multipleValidCallsReleaseAtomicallyOnceWithDistinctHostIDs() throws {
        let invocations = InvocationCounter()
        let tools = [tool(), tool("ping", properties: [], required: [])]
        var decoder = decoder(tools: tools, idGenerator: {
            "host-\(invocations.increment())"
        })
        _ = try decoder.consume(frame() + "\n" + frame(function: "ping", parameters: []))
        #expect(!decoder.hasToolCalls)
        let released = try decoder.finish()
        #expect(released.count == 2)
        #expect(released.compactMap { event -> String? in
            guard case let .toolCall(call) = event else { return nil }
            return call.id
        } == ["host-1", "host-2"])
        #expect(invocations.value == 2)
        #expect(try decoder.finish().isEmpty)
        #expect(invocations.value == 2)
    }

    @Test func thoughtIsSuppressedAndVisibleTextSurvivesMarkerSplits() throws {
        var decoder = decoder(startsInThoughtChannel: true)
        let text = "private 雪\nline</think>answer"
        var events: [StructuredAssistantEvent] = []
        for character in text {
            events += try decoder.consume(String(character))
        }
        let visible = events.compactMap { event -> String? in
            guard case let .content(text) = event else { return nil }
            return text
        }.joined()
        #expect(visible == "answer")
    }

    @Test func unicodeMultilineRawParameterSurvivesArbitrarySplits() throws {
        let value = "雪<&>'\nsecond\r\n"
        var decoder = decoder(tools: [tool()])
        for character in frame(parameters: [("query", value)]) {
            #expect(try decoder.consume(String(character)).isEmpty)
        }
        let events = try decoder.finish()
        guard case let .toolCall(call) = events.single else {
            Issue.record("expected one released Qwen call")
            return
        }
        #expect(call.arguments == .object(["query": .string(value)]))
        #expect(call.id == "host-id")
    }

    @Test func validCallThenMalformedTailPublishesNothingAndConsumesNoHostID() throws {
        let invocations = InvocationCounter()
        var decoder = decoder(tools: [tool()], idGenerator: {
            "host-\(invocations.increment())"
        })
        _ = try decoder.consume(frame())
        _ = try decoder.consume("\n<tool_call>\n<function=lookup>\n<parameter=query>\nunfinished")

        #expect(!decoder.hasToolCalls)
        #expect(invocations.value == 0)
        #expect(throws: GemmaToolCallParserError.self) {
            try decoder.finish()
        }
        #expect(invocations.value == 0)
        #expect(!decoder.hasToolCalls)
        #expect(throws: GemmaToolCallParserError.self) {
            try decoder.consume("later")
        }
    }

    @Test func incompleteEOFInsideEveryStructuredSpanFailsClosed() throws {
        let complete = Array(frame())
        for end in 1..<complete.count {
            let prefix = String(complete[..<end])
            let invocations = InvocationCounter()
            var decoder = decoder(
                tools: [tool()], idGenerator: countingGenerator(invocations))
            var consumeFailed = false
            do { _ = try decoder.consume(prefix) } catch { consumeFailed = true }
            var finishFailed = false
            do { _ = try decoder.finish() } catch { finishFailed = true }
            #expect(consumeFailed || finishFailed)
            #expect(!decoder.hasToolCalls)
            #expect(invocations.value == 0)
        }
    }

    @Test func everyTwoChunkSplitOfCompleteFrameReleasesExactlyOneCall() throws {
        let characters = Array(frame())
        for split in 1..<characters.count {
            var decoder = decoder(tools: [tool()])
            var streamed: [StructuredAssistantEvent] = []
            streamed += try decoder.consume(String(characters[..<split]))
            streamed += try decoder.consume(String(characters[split...]))
            #expect(streamed.isEmpty)
            let released = try decoder.finish()
            #expect(released.count == 1)
            #expect(decoder.hasToolCalls)
        }
    }

    @Test func everyTwoChunkSplitOfThoughtMarkersPreservesVisibleContent() throws {
        let text = "private</think>visible"
        let characters = Array(text)
        for split in 1..<characters.count {
            var decoder = decoder(startsInThoughtChannel: true)
            var events: [StructuredAssistantEvent] = []
            events += try decoder.consume(String(characters[..<split]))
            events += try decoder.consume(String(characters[split...]))
            let visible = events.compactMap { event -> String? in
                guard case let .content(value) = event else { return nil }
                return value
            }.joined()
            #expect(visible == "visible")
            _ = try decoder.finish()
        }
    }

    @Test func invalidUnknownAndNonWhitespaceTailNeverReleasesCalls() throws {
        let invalidInputs = [
            frame(function: "write"),
            frame() + "\nvisible text",
            frame() + "\n<tool_call>\n<function=write>\n</function>\n</tool_call>",
            "<tool_call>\n<function=lookup>\n<parameter=query>\nsnow</parameter>\n</function>\n</tool_call>",
        ]
        for input in invalidInputs {
            let invocations = InvocationCounter()
            var decoder = decoder(
                tools: [tool()], idGenerator: countingGenerator(invocations))
            var consumeFailed = false
            do { _ = try decoder.consume(input) } catch { consumeFailed = true }
            var finishFailed = false
            do { _ = try decoder.finish() } catch { finishFailed = true }
            #expect(consumeFailed || finishFailed)
            #expect(!decoder.hasToolCalls)
            #expect(invocations.value == 0)
        }
    }

    @Test func aggregatePendingCallBudgetFailsWithManySmallCalls() throws {
        let emptyTool = tool("ping", properties: [], required: [])
        let one = frame(function: "ping", parameters: []) + "\n"
        let input = String(repeating: one, count: 5_000)
        let invocations = InvocationCounter()
        var decoder = decoder(
            tools: [emptyTool], idGenerator: countingGenerator(invocations))
        #expect(throws: GemmaToolCallParserError.oversized) {
            try decoder.consume(input)
        }
        #expect(!decoder.hasToolCalls)
        #expect(invocations.value == 0)
        #expect(throws: GemmaToolCallParserError.self) {
            try decoder.finish()
        }
    }

    @Test func deeplyNestedArrayFailsTypedWithZeroHostIDs() throws {
        let depth = 16_384
        let body = String(repeating: "[", count: depth)
            + "0" + String(repeating: "]", count: depth)
        let nestedTool = tool(properties: [("value", .object([
            .init("type", .string("array")),
        ]))], required: ["value"])
        let invocations = InvocationCounter()
        var decoder = decoder(
            tools: [nestedTool], idGenerator: countingGenerator(invocations))
        var typedFailure = false
        do {
            _ = try decoder.consume(frame(parameters: [("value", body)]))
            _ = try decoder.finish()
        } catch let error as GemmaToolCallParserError {
            typedFailure = true
            #expect(error == .malformed || error == .oversized)
        }
        #expect(typedFailure)
        #expect(!decoder.hasToolCalls)
        #expect(invocations.value == 0)
    }

    @Test func deeplyNestedObjectFailsTypedWithZeroHostIDs() throws {
        let depth = 16_384
        let body = String(repeating: "{\"x\":", count: depth)
            + "null" + String(repeating: "}", count: depth)
        let nestedTool = tool(properties: [("value", .object([
            .init("type", .string("object")),
            .init("additionalProperties", .bool(false)),
        ]))], required: ["value"])
        let invocations = InvocationCounter()
        var decoder = decoder(
            tools: [nestedTool], idGenerator: countingGenerator(invocations))
        var typedFailure = false
        do {
            _ = try decoder.consume(frame(parameters: [("value", body)]))
            _ = try decoder.finish()
        } catch let error as GemmaToolCallParserError {
            typedFailure = true
            #expect(error == .malformed || error == .oversized)
        }
        #expect(typedFailure)
        #expect(!decoder.hasToolCalls)
        #expect(invocations.value == 0)
    }

    @Test func postFinishOperationsThrowAndSuccessfulFinishDoesNotDuplicate() throws {
        var decoder = decoder(tools: [tool()])
        _ = try decoder.consume(frame())
        _ = try decoder.finish()
        #expect(try decoder.finish().isEmpty)
        #expect(throws: GemmaToolCallParserError.self) {
            try decoder.consume("text")
        }
    }

    @Test func qwenEOSCompletesHeldFrameAllowsOneTailAndPublishesOnlyAtFinish() throws {
        let qwenDescriptor = try Self.qwenDescriptor()
        let characters = Array(frame())
        let endMarker = "</tool_call>"
        let prefix = String(characters.dropLast(endMarker.count))

        let valid = try StructuredAssistantDecoder(
            descriptor: qwenDescriptor,
            qwenTools: [tool()],
            startsInThoughtChannel: false,
            idGenerator: { "eos-host-id" })
        #expect(try valid.consume(tokenID: 0, delta: prefix).isEmpty)
        #expect(!valid.hasToolCalls)
        // 248046 is the independently pinned Qwen <|im_end|> token. The
        // detokenizer-held suffix completes the pending frame here.
        #expect(try valid.consume(
            tokenID: Int32(248_046), delta: endMarker + "\n").isEmpty)
        #expect(!valid.hasToolCalls)
        #expect(try valid.consumeTail("\n").isEmpty)
        #expect(try valid.finish() == [.toolCall(ParsedToolCall(
            id: "eos-host-id",
            name: "lookup",
            arguments: .object(["query": .string("snow")]),
            argumentsJSON: "{\"query\":\"snow\"}"))])
        #expect(try valid.finish().isEmpty)

        let postEOSInvocations = InvocationCounter()
        let postEOS = try StructuredAssistantDecoder(
            descriptor: qwenDescriptor,
            qwenTools: [tool()],
            startsInThoughtChannel: false,
            idGenerator: countingGenerator(postEOSInvocations))
        _ = try postEOS.consume(tokenID: 0, delta: frame())
        _ = try postEOS.consume(tokenID: Int32(248_046), delta: "\n")
        #expect(throws: GemmaToolCallParserError.self) {
            try postEOS.consume(tokenID: 0, delta: "after-eos")
        }
        #expect(throws: GemmaToolCallParserError.self) {
            try postEOS.finish()
        }
        #expect(postEOSInvocations.value == 0)
        #expect(!postEOS.hasToolCalls)

        let secondTailInvocations = InvocationCounter()
        let secondTail = try StructuredAssistantDecoder(
            descriptor: qwenDescriptor,
            qwenTools: [tool()],
            startsInThoughtChannel: false,
            idGenerator: countingGenerator(secondTailInvocations))
        _ = try secondTail.consume(tokenID: 0, delta: frame())
        _ = try secondTail.consume(tokenID: Int32(248_046), delta: "\n")
        #expect(try secondTail.consumeTail("\n").isEmpty)
        #expect(throws: GemmaToolCallParserError.self) {
            try secondTail.consumeTail("\n")
        }
        #expect(throws: GemmaToolCallParserError.self) {
            try secondTail.finish()
        }
        #expect(secondTailInvocations.value == 0)
        #expect(!secondTail.hasToolCalls)

        let malformedInvocations = InvocationCounter()
        let malformed = try StructuredAssistantDecoder(
            descriptor: qwenDescriptor,
            qwenTools: [tool()],
            startsInThoughtChannel: false,
            idGenerator: countingGenerator(malformedInvocations))
        _ = try malformed.consume(tokenID: 0, delta: frame())
        _ = try malformed.consume(tokenID: Int32(248_046), delta: "\n")
        #expect(throws: GemmaToolCallParserError.self) {
            try malformed.consumeTail("not whitespace")
        }
        #expect(throws: GemmaToolCallParserError.self) {
            try malformed.finish()
        }
        #expect(malformedInvocations.value == 0)
        #expect(!malformed.hasToolCalls)
    }

    @Test func descriptorFamilyMismatchFailsWithoutTryingAnotherGrammar() throws {
        let gemmaDescriptor = try Self.gemmaDescriptor()
        #expect(throws: GemmaToolCallParserError.malformed) {
            try StructuredAssistantDecoder(
                descriptor: gemmaDescriptor,
                qwenTools: [tool()],
                startsInThoughtChannel: false)
        }
    }

    private func modelChatValue(from value: JSONValue) throws -> ModelChatJSONValue {
        switch value {
        case .object(let members):
            return .object(try members.keys.sorted().map { key in
                guard let nested = members[key] else {
                    throw GemmaToolCallParserError.malformed
                }
                return .init(key, try modelChatValue(from: nested))
            })
        case .array(let values):
            return .array(try values.map(modelChatValue(from:)))
        case .string(let value): return .string(value)
        case .integer(let value): return .integer(value)
        case .unsignedInteger(let value): return .unsignedInteger(value)
        case .decimal(let value):
            let double = NSDecimalNumber(decimal: value).doubleValue
            guard double.isFinite,
                  let roundTrip = Decimal(
                    string: String(double), locale: Locale(identifier: "en_US_POSIX")),
                  roundTrip == value else {
                throw GemmaToolCallParserError.malformed
            }
            return .number(double)
        case .number(let value): return .number(value)
        case .bool(let value): return .bool(value)
        case .null: return .null
        }
    }

    private func localQwenCodec() throws -> QwenChatCodec {
        let source = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(
                "scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0",
                isDirectory: true)
        let evidence = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(
                "scratch/qwen3.6-35b-a3b/evidence/phase-15", isDirectory: true)
        let directory = evidence.appendingPathComponent(
            "qwen-roundtrip-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            for name in ["tokenizer.json", "tokenizer_config.json"] {
                let bytes = try Data(
                    contentsOf: source.appendingPathComponent(name), options: [.mappedIfSafe])
                try bytes.write(
                    to: directory.appendingPathComponent(name), options: [.atomic])
            }
            let tokenizer = try QwenTokenizer.loadOfficialSidecar(from: directory)
            return QwenChatCodec(tokenizer: tokenizer)
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    private static func qwenDescriptor() throws -> InstalledModelDescriptor {
        let digest = String(repeating: "a", count: 64)
        let layers: [GTurboQwenLayerTypeV2] = (0..<40).map {
            ($0 + 1).isMultiple(of: 4) ? .fullAttention : .linearAttention
        }
        let architecture = GTurboQwenArchitectureV2(
            hiddenSize: 2_048, numLayers: 40, layerTypes: layers,
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
        let groups = GTurboQuantizationCategoryV2.allCases.map { category in
            category == .recurrentState
                ? GTurboQuantizationGroupV2(category: category, storage: .fp32)
                : GTurboQuantizationGroupV2(
                    category: category, storage: .affineInt4, groupSize: 64,
                    scaleType: "bf16", biasType: "bf16")
        }
        let omissions = [
            "fc.weight", "layers.0.input_layernorm.weight",
            "layers.0.mlp.experts.down_proj", "layers.0.mlp.experts.gate_up_proj",
            "layers.0.mlp.gate.weight", "layers.0.mlp.shared_expert.down_proj.weight",
            "layers.0.mlp.shared_expert.gate_proj.weight",
            "layers.0.mlp.shared_expert.up_proj.weight",
            "layers.0.mlp.shared_expert_gate.weight",
            "layers.0.post_attention_layernorm.weight", "layers.0.self_attn.k_norm.weight",
            "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.o_proj.weight",
            "layers.0.self_attn.q_norm.weight", "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.v_proj.weight", "norm.weight",
            "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
        ].map { GTurboIgnoredTensorV2(name: "mtp." + $0, reason: .unsupportedMTP) }
        let manifest = GTurboManifestV2(
            family: .qwen3_6,
            requiredFeatures: [.familyDispatch, .verifiedIdentity, .qwenHybridAttention, .qwenMTPExcluded],
            modelID: GTurboFormatV2.qwenRepository,
            architecture: .qwen3_6(architecture),
            provenance: .init(
                sourceRepository: GTurboFormatV2.qwenRepository,
                sourceRevision: GTurboFormatV2.qwenRevision,
                sourceIndexSHA256: GTurboFormatV2.qwenSourceIndexSHA256,
                sidecarSHA256: GTurboFormatV2.qwenSidecarSHA256,
                quantizationPolicySHA256: digest),
            quantization: groups, ignoredTensors: omissions,
            files: [
                "model_weights.bin": .init(size: 32_768, sha256: digest),
                "packed_experts/layout.json": .init(size: 1, sha256: digest),
            ],
            tensorRegions: [.init(
                name: "embed", file: "model_weights.bin", offset: 0, size: 16,
                shape: [1], storage: .affineInt4,
                quantizationCategory: .embedding)],
            expertsPerLayer: 256, numLayers: 40, expertStride: 16_384)
        let data = try GTurboManifestV2Codec.encode(manifest)
        return try ManifestReader.decodeVerified(data: data).descriptor
    }

    private static func gemmaDescriptor() throws -> InstalledModelDescriptor {
        let digest = String(repeating: "a", count: 64)
        let architecture = GTurboManifestArchV1(
            hiddenSize: 64, ffnIntermediate: 128, moeIntermediateSize: 32,
            numHeads: 4, numKVHeads: 2, numFullKVHeads: 1, headDim: 16,
            fullHeadDim: 32, vocabSize: 1_024, slidingWindow: 128,
            finalLogitSoftcap: 30, ropeTheta: 10_000, fullRopeTheta: 1_000_000,
            partialRotaryFactor: 0.25, numLayers: 1, numExperts: 2,
            topKExperts: 1, tieWordEmbeddings: true, attentionKEqV: true,
            hiddenActivation: "gelu_pytorch_tanh", fullAttentionLayerMask: [0])
        let categories: [GTurboQuantizationCategoryV2] = [
            .embedding, .attention, .router, .sharedExpert, .routedExpert,
        ]
        let groups = categories.map {
            GTurboQuantizationGroupV2(
                category: $0, storage: .affineInt4, groupSize: 64,
                scaleType: "bf16", biasType: "bf16")
        }
        let manifest = GTurboManifestV2(
            family: .gemma4,
            requiredFeatures: [.familyDispatch, .verifiedIdentity],
            modelID: "fixture/gemma", architecture: .gemma4(architecture),
            provenance: .init(
                sourceRepository: "fixture/gemma", sourceRevision: "fixture-revision",
                sourceIndexSHA256: digest, sidecarSHA256: ["config.json": digest],
                quantizationPolicySHA256: digest),
            quantization: groups, ignoredTensors: [],
            files: [
                "model_weights.bin": .init(size: 16_384, sha256: digest),
                "packed_experts/layout.json": .init(size: 1, sha256: digest),
            ],
            tensorRegions: [.init(
                name: "embed", file: "model_weights.bin", offset: 0, size: 16,
                shape: [1], storage: .affineInt4,
                quantizationCategory: .embedding)],
            expertsPerLayer: 2, numLayers: 1, expertStride: 16_384)
        let data = try GTurboManifestV2Codec.encode(manifest)
        return try ManifestReader.decodeVerified(data: data).descriptor
    }
}

private extension Array where Element == StructuredAssistantEvent {
    var single: Element? { count == 1 ? first : nil }
}
