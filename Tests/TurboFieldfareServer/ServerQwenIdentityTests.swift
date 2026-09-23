import Foundation
import CoreGraphics
import ImageIO
import Metal
import Testing
import UniformTypeIdentifiers
@testable import TurboFieldfare
@testable import TurboFieldfareFormat
@testable import TurboFieldfareServerCore

@Suite("Server Qwen identity", .serialized)
struct ServerQwenIdentityTests {
    @Test func serverArgumentsKeepGemmaDefaultAndCaptureAnOptionalAssertion() throws {
        let defaults = try ServerArguments.parse(["--model", "model.gturbo"])
        #expect(defaults.modelIDAssertion == nil)
        #expect(defaults.modelID == ServerArguments.legacyDefaultModelID)

        let qwenID = QwenServerFixture.modelID
        let asserted = try ServerArguments.parse([
            "--model", "qwen.gturbo", "--model-id", qwenID,
        ])
        #expect(asserted.modelIDAssertion == qwenID)
        #expect(asserted.modelID == qwenID)
    }

    @Test func serverModelIdentityUsesGemmaDefaultAndRequiresAnExactQwenID() throws {
        let gemma = try ServerModelIdentity.resolve(
            admission: .init(family: .gemma4, verifiedIdentity: nil),
            assertedModelID: nil)
        #expect(gemma.apiModelID == ServerArguments.legacyDefaultModelID)
        #expect(gemma.family == .gemma4)
        #expect(gemma.sourceRevision == nil)
        #expect(gemma.verifiedIdentity == nil)

        let textModel = try QwenServerFixture.makeTextModel()
        defer { try? FileManager.default.removeItem(at: textModel) }
        let admission = try ModelFamilyGenerationSession.inspect(
            directoryURL: textModel)
        let qwen = try #require(admission.verifiedIdentity)

        let resolved = try ServerModelIdentity.resolve(
            admission: admission, assertedModelID: QwenServerFixture.modelID)
        #expect(resolved.apiModelID == QwenServerFixture.modelID)
        #expect(resolved.family == .qwen3_6)
        #expect(resolved.sourceRevision == QwenServerFixture.revision)
        #expect(resolved.verifiedIdentity == qwen)

        let derived = try ServerModelIdentity.resolve(
            admission: admission, assertedModelID: nil)
        #expect(derived.apiModelID == QwenServerFixture.modelID)

        #expect(throws: ServerArgumentError.self) {
            try ServerModelIdentity.resolve(
                admission: admission, assertedModelID: "qwen3.6-wrong-id")
        }
        #expect(throws: ServerArgumentError.self) {
            try ServerModelIdentity.resolve(
                admission: .init(family: .qwen3_6, verifiedIdentity: nil),
                assertedModelID: QwenServerFixture.modelID)
        }
    }

    @Test func loopbackModelsAndCompletionExposeTheSameQwenIdentity() async throws {
        let backend = QwenIdentityHTTPBackend()
        let server = TurboFieldfareHTTPServer(
            modelID: QwenServerFixture.modelID,
            queueLimit: 1,
            backend: backend,
            visionCapability: "missing",
            modelFamily: .qwen3_6,
            modelRevision: QwenServerFixture.revision)
        let channel = try await server.start(port: 0)
        do {
            let port = try #require(channel.localAddress?.port)
            #expect(channel.localAddress?.ipAddress == "127.0.0.1")
            let (modelData, modelResponse) = try await QwenServerFixture.data(
                for: URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/models")!))
            #expect((modelResponse as? HTTPURLResponse)?.statusCode == 200)
            let models = try JSONDecoder().decode(OpenAIModelList.self, from: modelData)
            let model = try #require(models.data.first)
            #expect(model.id == QwenServerFixture.modelID)
            #expect(model.family == LoadedRuntimeFamily.qwen3_6.rawValue)
            #expect(model.revision == QwenServerFixture.revision)
            #expect(model.capabilities == ["text"])

            var request = URLRequest(
                url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "content-type")
            request.httpBody = Data(#"{"model":"Qwen/Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hello"}]}"#.utf8)
            let (completionData, completionResponse) = try await QwenServerFixture.data(
                for: request)
            #expect((completionResponse as? HTTPURLResponse)?.statusCode == 200)
            let object = try #require(
                JSONSerialization.jsonObject(with: completionData) as? [String: Any])
            #expect(object["model"] as? String == QwenServerFixture.modelID)
            let choices = try #require(object["choices"] as? [[String: Any]])
            let message = try #require(choices.first?["message"] as? [String: Any])
            #expect(message["content"] as? String == "from fake Qwen")
            try await server.shutdown()
        } catch {
            try? await server.shutdown()
            throw error
        }

        let streamingServer = TurboFieldfareHTTPServer(
            modelID: QwenServerFixture.modelID,
            queueLimit: 1,
            backend: QwenIdentityHTTPBackend(emitToolCall: true),
            visionCapability: "missing",
            modelFamily: .qwen3_6,
            modelRevision: QwenServerFixture.revision)
        let streamingChannel = try await streamingServer.start(port: 0)
        do {
            let streamingPort = try #require(streamingChannel.localAddress?.port)
            var streamingRequest = URLRequest(
                url: URL(string: "http://127.0.0.1:\(streamingPort)/v1/chat/completions")!)
            streamingRequest.httpMethod = "POST"
            streamingRequest.setValue("application/json", forHTTPHeaderField: "content-type")
            streamingRequest.httpBody = Data(#"{"model":"Qwen/Qwen3.6-35B-A3B","messages":[{"role":"user","content":"call lookup"}],"stream":true,"stream_options":{"include_usage":true}}"#.utf8)
            let (streamData, streamResponse) = try await QwenServerFixture.data(
                for: streamingRequest)
            #expect((streamResponse as? HTTPURLResponse)?.statusCode == 200)
            let stream = String(decoding: streamData, as: UTF8.self)
            #expect(stream.contains(#""model":"Qwen\/Qwen3.6-35B-A3B""#))
            #expect(stream.contains(#""tool_calls""#))
            #expect(stream.contains(#""name":"lookup""#))
            #expect(stream.contains(#""finish_reason":"tool_calls""#))
            #expect(stream.contains(#""cached_tokens":0"#))
            #expect(stream.hasSuffix("data: [DONE]\n\n"))
            try await streamingServer.shutdown()
        } catch {
            try? await streamingServer.shutdown()
            throw error
        }
    }

    @Test func textOnlyLoadedIdentityAcceptsAValidatedCompanion() throws {
        let textModel = try QwenServerFixture.makeTextModel()
        defer { try? FileManager.default.removeItem(at: textModel) }
        let admission = try ModelFamilyGenerationSession.inspect(
            directoryURL: textModel)
        let loadedIdentity = try #require(admission.verifiedIdentity)
        let companion = try QwenServerFixture.makeSyntheticCompanion(
            beside: textModel, textManifestSHA256: loadedIdentity.textManifestSHA256)
        defer { try? FileManager.default.removeItem(at: companion) }

        let status = try ModelFamilyGenerationSession.inspectQwenVisionCompanion(
            directoryURL: textModel, loadedIdentity: loadedIdentity)
        #expect(status == expectedReadyOrUnsupported())
    }

    @Test func companionStatusReportsMissingAndHonorsAnExplicitMissingPath() throws {
        let textModel = try QwenServerFixture.makeTextModel()
        defer { try? FileManager.default.removeItem(at: textModel) }
        let admission = try ModelFamilyGenerationSession.inspect(
            directoryURL: textModel)
        let loadedIdentity = try #require(admission.verifiedIdentity)

        let missing = try ModelFamilyGenerationSession.inspectQwenVisionCompanion(
            directoryURL: textModel, loadedIdentity: loadedIdentity)
        #expect(missing == .missing)

        let explicitPath = textModel.deletingLastPathComponent()
            .appendingPathComponent("explicit-missing.vision.gturbo")
        #expect(throws: VisionPackError.self) {
            try ModelFamilyGenerationSession.inspectQwenVisionCompanion(
                directoryURL: textModel,
                loadedIdentity: loadedIdentity,
                visionPackURL: explicitPath)
        }
    }

    @Test func malformedCompanionIsInvalidAndCannotAdvertiseReady() throws {
        let textModel = try QwenServerFixture.makeTextModel()
        defer { try? FileManager.default.removeItem(at: textModel) }
        let admission = try ModelFamilyGenerationSession.inspect(
            directoryURL: textModel)
        let loadedIdentity = try #require(admission.verifiedIdentity)
        let companion = textModel.deletingLastPathComponent()
            .appendingPathComponent("\(textModel.deletingPathExtension().lastPathComponent).vision.gturbo")
        try FileManager.default.createDirectory(
            at: companion, withIntermediateDirectories: true)
        try Data("not-json".utf8).write(
            to: companion.appendingPathComponent("manifest.json"))
        defer { try? FileManager.default.removeItem(at: companion) }

        let status = try ModelFamilyGenerationSession.inspectQwenVisionCompanion(
            directoryURL: textModel, loadedIdentity: loadedIdentity)
        #expect(status == .invalid)
    }

    @Test func companionIdentityMismatchIsRejectedBeforeItsMetadataIsTrusted() throws {
        let textModel = try QwenServerFixture.makeTextModel()
        defer { try? FileManager.default.removeItem(at: textModel) }
        let admission = try ModelFamilyGenerationSession.inspect(
            directoryURL: textModel)
        let baseIdentity = try #require(admission.verifiedIdentity)
        let mismatched = try QwenServerFixture.identityWithChangedTextManifest(
            basedOn: baseIdentity,
            textManifestSHA256: String(repeating: "f", count: 64))

        #expect(throws: ModelFamilyGenerationError.modelIdentityChanged) {
            try ModelFamilyGenerationSession.inspectQwenVisionCompanion(
                directoryURL: textModel, loadedIdentity: mismatched)
        }
    }

    @Test func externallyDeclaredVisionMustMatchTheOpenedCompanion() throws {
        let textModel = try QwenServerFixture.makeTextModel()
        defer { try? FileManager.default.removeItem(at: textModel) }
        let admission = try ModelFamilyGenerationSession.inspect(
            directoryURL: textModel)
        let baseIdentity = try #require(admission.verifiedIdentity)
        let companion = try QwenServerFixture.makeSyntheticCompanion(
            beside: textModel, textManifestSHA256: baseIdentity.textManifestSHA256)
        defer { try? FileManager.default.removeItem(at: companion) }

        let matching = try QwenServerFixture.identityWithVision(
            basedOn: baseIdentity, companion: companion)
        let matchingStatus = try ModelFamilyGenerationSession.inspectQwenVisionCompanion(
            directoryURL: textModel, loadedIdentity: matching)
        #expect(matchingStatus == expectedReadyOrUnsupported())

        let stale = try QwenServerFixture.identityWithVision(
            basedOn: baseIdentity,
            companion: companion,
            visionPayloadSHA256: String(repeating: "d", count: 64))
        #expect(try ModelFamilyGenerationSession.inspectQwenVisionCompanion(
            directoryURL: textModel, loadedIdentity: stale) == .invalid)
    }

    @Test func corruptCompanionIsInvalidBeforeHardwareStatus() throws {
        let textModel = try QwenServerFixture.makeTextModel()
        defer { try? FileManager.default.removeItem(at: textModel) }
        let loadedIdentity = try #require(
            ModelFamilyGenerationSession.inspect(directoryURL: textModel).verifiedIdentity)
        let companion = try QwenServerFixture.makeSyntheticCompanion(
            beside: textModel, textManifestSHA256: loadedIdentity.textManifestSHA256)
        defer { try? FileManager.default.removeItem(at: companion) }
        let weights = companion.appendingPathComponent(GTurboVisionFormatV2.weightsFile)
        let handle = try FileHandle(forWritingTo: weights)
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: Data([1]))
        try handle.close()

        let status = try ModelFamilyGenerationSession.inspectQwenVisionCompanion(
            directoryURL: textModel, loadedIdentity: loadedIdentity)
        #expect(status == .invalid)
    }

    @Test(.enabled(if: VisionRuntime.isSupportedOnDefaultDevice,
                   "requires the verified Qwen image hardware gate"))
    func preflightImagesAcceptsTextOnlyIdentityWithAValidatedCompanion() throws {
        let textModel = try QwenServerFixture.makeTextModel(includeTokenizer: true)
        defer { try? FileManager.default.removeItem(at: textModel) }
        let loadedIdentity = try #require(
            ModelFamilyGenerationSession.inspect(directoryURL: textModel).verifiedIdentity)
        let companion = try QwenServerFixture.makeSyntheticCompanion(
            beside: textModel, textManifestSHA256: loadedIdentity.textManifestSHA256)
        defer { try? FileManager.default.removeItem(at: companion) }
        let image = try QwenServerFixture.makeSolidImage()
        defer { try? FileManager.default.removeItem(at: image.deletingLastPathComponent()) }

        let result = try ModelFamilyGenerationSession.preflightQwen(
            directoryURL: textModel,
            prompt: .chat(messages: [
                .init(role: .user, content: .parts([
                    .text("describe"), .image(.init(id: "image-1")),
                ])),
            ], tools: [], thinking: .automatic),
            imagesByID: ["image-1": image],
            visionPackURL: companion,
            visionResidency: .onDemand,
            maxContext: 4_096)
        #expect(result.imageCount == 1)
        #expect(result.promptTokens > 0)
    }

    @Test func unsupportedCompanionStatusIsOnlyReturnedForAValidPack() throws {
        let textModel = try QwenServerFixture.makeTextModel()
        defer { try? FileManager.default.removeItem(at: textModel) }
        let loadedIdentity = try #require(
            ModelFamilyGenerationSession.inspect(directoryURL: textModel).verifiedIdentity)
        let companion = try QwenServerFixture.makeSyntheticCompanion(
            beside: textModel, textManifestSHA256: loadedIdentity.textManifestSHA256)
        defer { try? FileManager.default.removeItem(at: companion) }
        let status = try ModelFamilyGenerationSession.inspectQwenVisionCompanion(
            directoryURL: textModel, loadedIdentity: loadedIdentity)
        #expect(status == expectedReadyOrUnsupported())
    }
}

private actor QwenIdentityHTTPBackend: ServerInferenceBackend {
    nonisolated let requestFamily = LoadedRuntimeFamily.qwen3_6
    private let emitToolCall: Bool

    init(emitToolCall: Bool = false) {
        self.emitToolCall = emitToolCall
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        if emitToolCall {
            let call = ParsedToolCall(
                id: "call_000000000000000000000001",
                name: "lookup",
                arguments: .object(["query": .string("snow")]),
                argumentsJSON: #"{"query":"snow"}"#)
            onEvent(.toolCall(call))
            return ServerCompletion(
                content: "",
                toolCalls: [call],
                finishReason: "tool_calls",
                usage: OpenAIUsage(promptTokens: 2, completionTokens: 3, totalTokens: 5))
        }
        onEvent(.content("from fake Qwen"))
        return ServerCompletion(
            content: "from fake Qwen",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 2, completionTokens: 3, totalTokens: 5))
    }
}

private enum QwenServerFixture {
    static let modelID = "Qwen/Qwen3.6-35B-A3B"
    static let revision = "995ad96eacd98c81ed38be0c5b274b04031597b0"
    static let sourceIndexSHA256 = "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
    static let processorConfigSHA256 = "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    static let sidecars: [String: String] = [
        "config.json": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
        "configuration.json": "c1b09db419119513247e9b8b912c4b9897106c9b20c6cada7e107d993c5435eb",
        "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
        "model.safetensors.index.json": sourceIndexSHA256,
        "preprocessor_config.json": processorConfigSHA256,
        "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "tokenizer_config.json": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
    ]

    static func makeTextModel(includeTokenizer: Bool = false) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("server-qwen-\(UUID().uuidString).gturbo")
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        try manifestData().write(to: root.appendingPathComponent("manifest.json"))
        if includeTokenizer {
            let tokenizerDirectory = root.appendingPathComponent(
                "tokenizer", isDirectory: true)
            try FileManager.default.createDirectory(
                at: tokenizerDirectory, withIntermediateDirectories: true)
            let source = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(
                    "scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0")
            try FileManager.default.copyItem(
                at: source.appendingPathComponent("tokenizer.json"),
                to: tokenizerDirectory.appendingPathComponent("tokenizer.json"))
            try FileManager.default.copyItem(
                at: source.appendingPathComponent("tokenizer_config.json"),
                to: tokenizerDirectory.appendingPathComponent("tokenizer_config.json"))
        }
        return root
    }

    static func manifestData(
        textManifestSHA256: String = String(repeating: "a", count: 64)
    ) throws -> Data {
        let layerTypes = (0..<40).map { index in
            (index + 1).isMultiple(of: 4) ? "fullAttention" : "linearAttention"
        }
        let architecture: [String: Any] = [
            "family": "qwen3_6",
            "configuration": [
                "hiddenSize": 2048,
                "numLayers": 40,
                "layerTypes": layerTypes,
                "numAttentionHeads": 16,
                "numKeyValueHeads": 2,
                "headDimension": 256,
                "attentionOutputGate": true,
                "linearConvolutionKernel": 4,
                "linearKeyHeads": 16,
                "linearKeyHeadDimension": 128,
                "linearValueHeads": 32,
                "linearValueHeadDimension": 128,
                "recurrentStateType": "fp32",
                "partialRotaryFactor": 0.25,
                "ropeTheta": 10_000_000,
                "mropeInterleaved": true,
                "mropeSections": [11, 11, 10],
                "numberOfExperts": 256,
                "expertsPerToken": 8,
                "routedExpertIntermediateSize": 512,
                "sharedExpertIntermediateSize": 512,
                "vocabularySize": 248_320,
                "tiedWordEmbeddings": false,
                "hiddenActivation": "silu",
                "bosTokenID": 248_044,
                "eosTokenID": 248_044,
                "imageTokenID": 248_056,
                "videoTokenID": 248_057,
                "visionStartTokenID": 248_053,
                "visionEndTokenID": 248_054,
            ],
        ]
        let categories = [
            "embedding", "attention", "linearAttention", "router",
            "sharedExpert", "routedExpert", "outputHead", "normalization",
            "recurrentState",
        ]
        let quantization: [[String: Any]] = categories.map { category in
            if category == "recurrentState" {
                return ["category": category, "storage": "fp32"]
            }
            return [
                "category": category,
                "storage": "affineInt4",
                "groupSize": 64,
                "scaleType": "bf16",
                "biasType": "bf16",
            ]
        }
        let ignored = [
            "fc.weight", "layers.0.input_layernorm.weight",
            "layers.0.mlp.experts.down_proj", "layers.0.mlp.experts.gate_up_proj",
            "layers.0.mlp.gate.weight", "layers.0.mlp.shared_expert.down_proj.weight",
            "layers.0.mlp.shared_expert.gate_proj.weight",
            "layers.0.mlp.shared_expert.up_proj.weight",
            "layers.0.mlp.shared_expert_gate.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.0.self_attn.k_norm.weight", "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.o_proj.weight", "layers.0.self_attn.q_norm.weight",
            "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.v_proj.weight",
            "norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
        ].map { ["name": "mtp.\($0)", "reason": "unsupportedMTP"] }
        let sha = String(repeating: "b", count: 64)
        let root: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 2,
            "versionMinor": 0,
            "family": "qwen3_6",
            "requiredFeatures": [
                "familyDispatch", "verifiedIdentity", "qwenHybridAttention", "qwenMTPExcluded",
            ],
            "modelID": modelID,
            "architecture": architecture,
            "provenance": [
                "sourceRepository": modelID,
                "sourceRevision": revision,
                "sourceIndexSHA256": sourceIndexSHA256,
                "sidecarSHA256": sidecars,
                "quantizationPolicySHA256": textManifestSHA256,
            ],
            "quantization": quantization,
            "ignoredTensors": ignored,
            "files": [
                "model_weights.bin": ["size": 16, "sha256": sha],
                "packed_experts/layout.json": ["size": 1, "sha256": sha],
            ],
            "tensorRegions": [[
                "name": "embed", "file": "model_weights.bin", "offset": 0,
                "size": 16, "shape": [1], "storage": "affineInt4",
                "quantizationCategory": "embedding",
            ]],
            "expertsPerLayer": 256,
            "numLayers": 40,
            "expertStride": 16_384,
        ]
        return try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
    }

    static func data(for request: URLRequest) async throws -> (Data, URLResponse) {
        var bounded = request
        bounded.timeoutInterval = 5
        return try await URLSession.shared.data(for: bounded)
    }

    static func identityWithChangedTextManifest(
        basedOn base: LoadedRuntimeIdentity,
        textManifestSHA256: String
    ) throws -> LoadedRuntimeIdentity {
        let vision = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(InstalledVisionStatus.unavailable))
        return try makeIdentity(
            basedOn: base,
            textManifestSHA256: textManifestSHA256,
            vision: vision)
    }

    static func identityWithVision(
        basedOn base: LoadedRuntimeIdentity,
        companion: URL,
        visionPayloadSHA256: String? = nil
    ) throws -> LoadedRuntimeIdentity {
        let manifest = try JSONSerialization.jsonObject(
            with: Data(contentsOf: companion.appendingPathComponent("manifest.json")))
        let object = try #require(manifest as? [String: Any])
        let actualPayload = try #require(object["visionPayloadSHA256"] as? String)
        return try makeIdentity(
            basedOn: base,
            textManifestSHA256: base.textManifestSHA256,
            vision: ["verified": ["_0": [
                "sourceRevision": base.sourceRevision,
                "processorConfigSHA256": processorConfigSHA256,
                "compatibleTextManifestSHA256": base.textManifestSHA256,
                "visionPayloadSHA256": visionPayloadSHA256 ?? actualPayload,
                "supportsStillImages": true,
                "supportsVideo": false,
            ]]])
    }

    private static func makeIdentity(
        basedOn base: LoadedRuntimeIdentity,
        textManifestSHA256: String,
        vision: Any
    ) throws -> LoadedRuntimeIdentity {
        let quantization = base.quantization.map { group -> [String: Any] in
            var result: [String: Any] = [
                "category": group.category,
                "storage": group.storage,
            ]
            if let groupSize = group.groupSize { result["groupSize"] = groupSize }
            if let scaleType = group.scaleType { result["scaleType"] = scaleType }
            if let biasType = group.biasType { result["biasType"] = biasType }
            return result
        }
        let descriptorRoot: [String: Any] = [
            "family": base.family.rawValue,
            "modelID": base.modelID,
            "sourceRevision": base.sourceRevision,
            "formatMajor": base.formatMajor,
            "formatMinor": base.formatMinor,
            "sourceIndexSHA256": base.sourceIndexSHA256,
            "quantizationPolicySHA256": base.quantizationPolicySHA256,
            "textManifestSHA256": textManifestSHA256,
            "quantization": quantization,
            "vision": vision,
        ]
        let descriptorData = try JSONSerialization.data(
            withJSONObject: descriptorRoot, options: [])
        let descriptor = try JSONDecoder().decode(
            InstalledModelDescriptor.self, from: descriptorData)
        return LoadedRuntimeIdentity(descriptor: descriptor)
    }

    static func makeSyntheticCompanion(
        beside textModel: URL,
        textManifestSHA256: String
    ) throws -> URL {
        let companion = textModel.deletingLastPathComponent()
            .appendingPathComponent("\(textModel.deletingPathExtension().lastPathComponent).vision.gturbo")
        var keepRoot = false
        defer {
            if !keepRoot { try? FileManager.default.removeItem(at: companion) }
        }
        try FileManager.default.createDirectory(
            at: companion, withIntermediateDirectories: true)
        let processor = Data(
            """
            {
                "size": {
                    "longest_edge": 16777216,
                    "shortest_edge": 65536
                },
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "image_mean": [
                    0.5,
                    0.5,
                    0.5
                ],
                "image_std": [
                    0.5,
                    0.5,
                    0.5
                ],
                "processor_class": "Qwen3VLProcessor",
                "image_processor_type": "Qwen2VLImageProcessorFast"
            }
            """.utf8)
        // This is the pinned processor sidecar digest, and the payload below
        // is a sparse all-zero file with the official 333 tensor extents.
        // No authentic model weights are read by these identity tests.
        let processorDigest = Sha256Verifier.hashData(processor)
        #expect(processorDigest == processorConfigSHA256)
        try processor.write(to: companion.appendingPathComponent(
            GTurboVisionFormatV2.processorFile))

        var offset: UInt64 = 0
        var regions: [GTurboTensorRegionV2] = []
        for tensor in QwenVisionConfig.official.tensorContract {
            let alignment = UInt64(GTurboVisionFormatV2.alignmentBytes)
            let remainder = offset % alignment
            if remainder != 0 { offset += alignment - remainder }
            let elements = tensor.shape.reduce(UInt64(1), *)
            let size = elements * 2
            regions.append(.init(
                name: tensor.name,
                file: GTurboVisionFormatV2.weightsFile,
                offset: offset,
                size: size,
                shape: tensor.shape,
                storage: .bf16))
            offset += size
        }
        #expect(offset == 896_602_112)
        let weightsURL = companion.appendingPathComponent(
            GTurboVisionFormatV2.weightsFile)
        FileManager.default.createFile(atPath: weightsURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: weightsURL)
        try handle.seek(toOffset: offset - 1)
        try handle.write(contentsOf: Data([0]))
        try handle.close()

        let payloadSHA = "b52d991b5cf81b4efafdd821892277a711949aaf2a93669c4e549bcfd5cea21d"
        let files: [String: GTurboManifestFileV1] = [
            GTurboVisionFormatV2.weightsFile: .init(
                size: offset, sha256: payloadSHA),
            GTurboVisionFormatV2.processorFile: .init(
                size: UInt64(processor.count), sha256: processorDigest),
        ]
        let manifest = GTurboVisionManifestV2(
            family: .qwen3_6,
            modelID: GTurboFormatV2.qwenRepository,
            sourceRevision: GTurboFormatV2.qwenRevision,
            processorProfile: .init(
                processorClass: "Qwen3VLProcessor",
                imageProcessorType: "Qwen2VLImageProcessorFast",
                patchSize: 16, temporalPatchSize: 2, spatialMergeSize: 2),
            processorConfigSHA256: processorDigest,
            compatibleTextManifestSHA256: textManifestSHA256,
            visionPayloadSHA256: payloadSHA,
            supportsStillImages: true,
            supportsVideo: false,
            files: files,
            tensorRegions: regions)
        let manifestData = try GTurboVisionManifestV2Codec.encode(manifest)
        try manifestData.write(to: companion.appendingPathComponent(
            GTurboVisionFormatV1.manifestFile))
        let manifestDigest = Sha256Verifier.hashData(manifestData)
        let receipt = VerifiedInstallReceipt(
            manifestSha256: manifestDigest,
            modelDirectoryPath: companion.standardizedFileURL.path,
            sourceRepoID: GTurboFormatV2.qwenRepository,
            sourceRevision: GTurboFormatV2.qwenRevision,
            verificationTimestamp: "fixture",
            toolVersion: "fixture",
            files: [
                GTurboVisionFormatV1.manifestFile: .init(
                    size: UInt64(manifestData.count), sha256: manifestDigest),
                GTurboVisionFormatV2.weightsFile: .init(
                    size: offset, sha256: payloadSHA),
                GTurboVisionFormatV2.processorFile: .init(
                    size: UInt64(processor.count), sha256: processorDigest),
            ])
        try JSONEncoder().encode(receipt).write(to: companion.appendingPathComponent(
            VerifiedInstallReceiptReader.fileName))
        keepRoot = true
        return companion
    }

    static func makeSolidImage() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("server-qwen-image-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        let url = root.appendingPathComponent("input.png")
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: nil, width: 256, height: 256, bitsPerComponent: 8,
                bytesPerRow: 256 * 4, space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let destination = CGImageDestinationCreateWithURL(
                url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw QwenVisionError.allocationFailed(name: "test image")
        }
        context.setFillColor(CGColor(red: 0.25, green: 0.5, blue: 0.75, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 256, height: 256))
        guard let image = context.makeImage() else {
            throw QwenVisionError.allocationFailed(name: "test image")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw QwenVisionError.allocationFailed(name: "test image")
        }
        return url
    }
}

private func expectedReadyOrUnsupported() -> ModelFamilyVisionCompanionStatus {
    guard let device = MTLCreateSystemDefaultDevice(),
          VisionRuntime.isSupported(on: device) else { return .unsupported }
    return .ready
}
