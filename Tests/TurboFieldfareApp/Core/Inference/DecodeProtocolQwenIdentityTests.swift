import Foundation
import Metal
import Synchronization
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareAppCore
@testable import TurboFieldfareFormat
import TurboFieldfareDecodeProtocol

@Suite struct DecodeProtocolQwenIdentityTests {
    @Test func validatedDescriptorMappingPreservesEveryIdentityField() throws {
        let admitted = try ManifestReader.decodeVerified(data: qwenManifestData())
        let original = admitted.descriptor
        var root = try #require(JSONSerialization.jsonObject(
            with: JSONEncoder().encode(original)) as? [String: Any])
        let quantization = try #require(root["quantization"] as? [[String: Any]])
        root["quantization"] = Array(quantization.reversed())
        let processorDigest = try #require(
            GTurboFormatV2.qwenSidecarSHA256["preprocessor_config.json"])
        root["vision"] = ["verified": ["_0": [
            "sourceRevision": GTurboFormatV2.qwenRevision,
            "processorConfigSHA256": processorDigest,
            "compatibleTextManifestSHA256": original.textManifestSHA256,
            "visionPayloadSHA256": digest,
            "supportsStillImages": true,
            "supportsVideo": false,
        ]]]
        let synthesized = try JSONSerialization.data(withJSONObject: root)
        let descriptor = try JSONDecoder().decode(
            type(of: original), from: synthesized)
        let runtime = LoadedRuntimeIdentity(descriptor: descriptor)
        let wire = DecodeModelIdentity(runtimeIdentity: runtime)

        #expect(wire.family == .qwen3_6)
        #expect(wire.modelID == descriptor.modelID)
        #expect(wire.sourceRevision == descriptor.sourceRevision)
        #expect(wire.formatMajor == descriptor.formatMajor)
        #expect(wire.formatMinor == descriptor.formatMinor)
        #expect(wire.sourceIndexSHA256 == descriptor.sourceIndexSHA256)
        #expect(wire.quantizationPolicySHA256 == descriptor.quantizationPolicySHA256)
        #expect(wire.textManifestSHA256 == descriptor.textManifestSHA256)
        #expect(wire.quantization.count == descriptor.quantization.count)
        #expect(wire.quantization.map(\.category)
            == descriptor.quantization.map(\.category))
        #expect(wire.quantization.map(\.storage)
            == descriptor.quantization.map(\.storage))
        for (wireGroup, descriptorGroup) in zip(
            wire.quantization, descriptor.quantization) {
            #expect(wireGroup.groupSize == descriptorGroup.groupSize)
            #expect(wireGroup.scaleType == descriptorGroup.scaleType)
            #expect(wireGroup.biasType == descriptorGroup.biasType)
        }
        guard case let .verified(vision) = wire.vision else {
            Issue.record("validated vision identity was lost")
            return
        }
        #expect(vision.sourceRevision == GTurboFormatV2.qwenRevision)
        #expect(vision.processorConfigSHA256 == processorDigest)
        #expect(vision.compatibleTextManifestSHA256 == descriptor.textManifestSHA256)
        #expect(vision.visionPayloadSHA256 == digest)
        #expect(vision.supportsStillImages)
        #expect(!vision.supportsVideo)
    }

    @Test func admittedTinyQwenBundleInstallsCodecAndIdentity() async throws {
        let fileManager = FileManager.default
        let root = fileManager.temporaryDirectory
            .appendingPathComponent("p17-qwen-bundle-\(UUID().uuidString)", isDirectory: true)
        defer { try? fileManager.removeItem(at: root) }
        try fileManager.createDirectory(at: root, withIntermediateDirectories: false)
        let tokenizerDirectory = root.appendingPathComponent("tokenizer", isDirectory: true)
        try fileManager.createDirectory(
            at: tokenizerDirectory, withIntermediateDirectories: false)

        let manifestData = try qwenManifestData()
        let admitted = try ManifestReader.decodeVerified(data: manifestData)
        try manifestData.write(
            to: root.appendingPathComponent("manifest.json"), options: .atomic)
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let officialDirectory = repositoryRoot
            .appendingPathComponent("scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0")
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            try fileManager.copyItem(
                at: officialDirectory.appendingPathComponent(name),
                to: tokenizerDirectory.appendingPathComponent(name))
        }
        let tokenizer = try QwenTokenizer.load(from: root)
        let codec = QwenChatCodec(tokenizer: tokenizer)

        let fixtureURL = repositoryRoot
            .appendingPathComponent("Tests/TurboFieldfare/Core/QwenFixtures/qwen36-tiny-text-model-fixtures.json")
        let records = try QwenTextFixtureRecords.decode(
            jsonData: Data(contentsOf: fixtureURL))
        let context = try MetalContext()
        let model = try QwenTextModel.loadFixtureRecords(records, device: context.device)
        let bundle = LoadedModelFamilyBundle(
            runtime: .qwen(model), family: .qwen3_6,
            verifiedIdentity: LoadedRuntimeIdentity(descriptor: admitted.descriptor),
            qwenCodec: codec)
        let session = RealInferenceSession()
        let key = SessionLoadKey(
            directory: root, maxContext: 128,
            options: AppRuntimeOptions(), forceLogitsHead: false)
        let states = Mutex<[AppModelLoadState]>([])
        try await session.installLoadedFamily(
            bundle, key: key, context: context) { state in
                states.withLock { $0.append(state) }
            }

        let expectedIdentity = DecodeModelIdentity(
            runtimeIdentity: LoadedRuntimeIdentity(descriptor: admitted.descriptor))
        guard case let .qwen(identity) = await session.loadedModelReadiness else {
            Issue.record("installed Qwen bundle did not expose Qwen readiness")
            return
        }
        #expect(identity == expectedIdentity)
        #expect(states.withLock { states in
            states.contains { state in
                if case .loading(.preparingRunner) = state { return true }
                return false
            }
        })
        await session.unload()
        let initialOwnedCount = await session.lifecycleOwnedInstallationCount
        #expect(initialOwnedCount == 0)

        let retirementEntered = AwaitOnce()
        let releaseRetirement = AwaitOnce()
        let preparationEntered = AwaitOnce()
        let releasePreparation = AwaitOnce()
        let lifecycleStates = Mutex<[AppModelLoadState]>([])
        let lifecycleSession = RealInferenceSession(
            lifecycleCheckpoints: RealInferenceLifecycleCheckpoints(
                afterRetirement: {
                    retirementEntered.signal()
                    await releaseRetirement.wait()
                },
                afterPreparation: {
                    preparationEntered.signal()
                    await releasePreparation.wait()
                }))
        let install = Task {
            try await lifecycleSession.installLoadedFamily(
                bundle, key: key, context: context) { state in
                    lifecycleStates.withLock { $0.append(state) }
                }
        }
        await retirementEntered.wait()
        let retiredOwnedCount = await lifecycleSession.lifecycleOwnedInstallationCount
        #expect(retiredOwnedCount == 0)
        await #expect(throws: RealInferenceLifecycleError.lifecycleInProgress) {
            try await lifecycleSession.installLoadedFamily(
                bundle, key: key, context: context)
        }
        releaseRetirement.signal()
        await preparationEntered.wait()
        let preparedOwnedCount = await lifecycleSession.lifecycleOwnedInstallationCount
        #expect(preparedOwnedCount == 1)
        await #expect(throws: RealInferenceLifecycleError.lifecycleInProgress) {
            try await lifecycleSession.installLoadedFamily(
                bundle, key: key, context: context)
        }

        let firstUnload = Task { await lifecycleSession.unload() }
        var firstWaiterArrived = false
        for _ in 0..<10_000 {
            if await lifecycleSession.lifecycleUnloadWaiterCount == 1 {
                firstWaiterArrived = true
                break
            }
            await Task.yield()
        }
        #expect(firstWaiterArrived)
        let secondUnload = Task { await lifecycleSession.unload() }
        var secondWaiterArrived = false
        for _ in 0..<10_000 {
            if await lifecycleSession.lifecycleUnloadWaiterCount == 2 {
                secondWaiterArrived = true
                break
            }
            await Task.yield()
        }
        #expect(secondWaiterArrived)
        let teardownRequested = await lifecycleSession.lifecycleTeardownIsRequested
        #expect(teardownRequested)
        secondUnload.cancel()
        let waitersAfterCancellation = await lifecycleSession.lifecycleUnloadWaiterCount
        #expect(waitersAfterCancellation == 2)
        releasePreparation.signal()
        await #expect(throws: CancellationError.self) { try await install.value }
        await firstUnload.value
        await secondUnload.value
        let finalOwnedCount = await lifecycleSession.lifecycleOwnedInstallationCount
        #expect(finalOwnedCount == 0)
        #expect(lifecycleStates.withLock { states in
            !states.contains { state in state.isReady || state.isFailed }
        })
    }

    @Test func legacyReadyPayloadStillDecodesWithoutAnUnverifiedIdentity() throws {
        let generationID = UUID()
        var object = try #require(JSONSerialization.jsonObject(
            with: JSONEncoder().encode(DecodeServiceEvent(
                kind: .ready, generationID: generationID))) as? [String: Any])
        object.removeValue(forKey: "loadedFamily")
        object.removeValue(forKey: "loadID")
        object.removeValue(forKey: "modelIdentity")
        let legacy = try JSONSerialization.data(withJSONObject: object)
        let decoded = try JSONDecoder().decode(DecodeServiceEvent.self, from: legacy)
        #expect(decoded.kind == .ready)
        #expect(decoded.generationID == generationID)
        #expect(decoded.loadedFamily == nil)
        #expect(decoded.loadID == nil)
        #expect(decoded.modelIdentity == nil)
    }

    @Test func loadAttemptAndCancellationIdentityRoundTripTogether() throws {
        let requestID = UUID()
        let attemptID = UUID()
        let request = DecodeLoadRequest(
            modelPath: "/tmp/qwen", maxContextTokens: 128,
            requestID: requestID, attemptID: attemptID)
        let cancel = DecodeServiceCommand.cancelLoad(
            DecodeCancelLoadRequest(requestID: requestID, attemptID: attemptID))
        let decodedRequest = try JSONDecoder().decode(
            DecodeLoadRequest.self, from: JSONEncoder().encode(request))
        let decodedCommand = try JSONDecoder().decode(
            DecodeServiceCommand.self, from: JSONEncoder().encode(cancel))

        #expect(decodedRequest.requestID == requestID)
        #expect(decodedRequest.attemptID == attemptID)
        guard case let .cancelLoad(decodedCancel) = decodedCommand else {
            Issue.record("attempt-bound cancellation was not retained")
            return
        }
        #expect(decodedCancel.requestID == requestID)
        #expect(decodedCancel.attemptID == attemptID)
    }

    @Test func lifetimeProbeAndAcknowledgementRoundTripWithoutModelIdentity() throws {
        let nonce = UUID()
        let command = DecodeServiceCommand.lifetimeProbe(
            DecodeLifetimeProbe(nonce: nonce))
        let acknowledgement = DecodeServiceEvent(
            kind: .lifetimeAcknowledged, generationID: nonce,
            lifetimeNonce: nonce)
        let decodedCommand = try JSONDecoder().decode(
            DecodeServiceCommand.self, from: JSONEncoder().encode(command))
        let decodedAcknowledgement = try JSONDecoder().decode(
            DecodeServiceEvent.self, from: JSONEncoder().encode(acknowledgement))

        guard case let .lifetimeProbe(probe) = decodedCommand else {
            Issue.record("lifetime probe was not retained")
            return
        }
        #expect(probe.nonce == nonce)
        #expect(decodedAcknowledgement.kind == .lifetimeAcknowledged)
        #expect(decodedAcknowledgement.generationID == nonce)
        #expect(decodedAcknowledgement.lifetimeNonce == nonce)
        #expect(decodedAcknowledgement.loadedFamily == nil)
        #expect(decodedAcknowledgement.loadID == nil)
        #expect(decodedAcknowledgement.modelIdentity == nil)
        #expect(decodedAcknowledgement.conversationEpoch == nil)
    }

    @Test func legacyLoadAndCancellationTerminalRemainIdentityFree() throws {
        let requestID = UUID()
        let legacy = DecodeLoadRequest(
            modelPath: "/tmp/gemma", maxContextTokens: 128, requestID: requestID)
        let decodedLoad = try JSONDecoder().decode(
            DecodeLoadRequest.self, from: JSONEncoder().encode(legacy))
        let event = DecodeServiceEvent(
            kind: .loadCancelled, generationID: requestID,
            loadAttemptID: nil, loadedFamily: nil, loadID: nil,
            modelIdentity: nil, conversationEpoch: nil)
        let decodedEvent = try JSONDecoder().decode(
            DecodeServiceEvent.self, from: JSONEncoder().encode(event))

        #expect(decodedLoad.requestID == requestID)
        #expect(decodedLoad.attemptID == nil)
        #expect(decodedEvent.kind == .loadCancelled)
        #expect(decodedEvent.loadAttemptID == nil)
        #expect(decodedEvent.loadedFamily == nil)
        #expect(decodedEvent.loadID == nil)
        #expect(decodedEvent.modelIdentity == nil)
        #expect(decodedEvent.conversationEpoch == nil)
    }

    @Test func requestIdentityAndEpochRoundTripTogether() throws {
        let loadID = UUID()
        let epoch = UUID()
        let request = DecodeGenerationRequest(
            prompt: "hello", maxNewTokens: 4, maxContextTokens: 32,
            temperature: 0, generationID: UUID(), conversationEpoch: epoch,
            turnIndex: 2, loadID: loadID)
        let decoded = try JSONDecoder().decode(
            DecodeGenerationRequest.self,
            from: JSONEncoder().encode(request))
        #expect(decoded.loadID == loadID)
        #expect(decoded.conversationEpoch == epoch)
        #expect(decoded.turnIndex == 2)
        #expect(decoded.generationID == request.generationID)
    }

    private let digest = String(repeating: "a", count: 64)

    private func qwenManifestData() throws -> Data {
        let layers: [GTurboQwenLayerTypeV2] = (0..<40).map {
            ($0 + 1).isMultiple(of: 4) ? .fullAttention : .linearAttention
        }
        let architecture = GTurboQwenArchitectureV2(
            hiddenSize: 2048, numLayers: 40, layerTypes: layers,
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
        let quantization = GTurboQuantizationCategoryV2.allCases.map { category in
            category == .recurrentState
                ? GTurboQuantizationGroupV2(category: category, storage: .fp32)
                : GTurboQuantizationGroupV2(
                    category: category, storage: .affineInt4, groupSize: 64,
                    scaleType: "bf16", biasType: "bf16")
        }
        let ignored = ["fc.weight", "layers.0.input_layernorm.weight",
                       "layers.0.mlp.experts.down_proj",
                       "layers.0.mlp.experts.gate_up_proj",
                       "layers.0.mlp.gate.weight",
                       "layers.0.mlp.shared_expert.down_proj.weight",
                       "layers.0.mlp.shared_expert.gate_proj.weight",
                       "layers.0.mlp.shared_expert.up_proj.weight",
                       "layers.0.mlp.shared_expert_gate.weight",
                       "layers.0.post_attention_layernorm.weight",
                       "layers.0.self_attn.k_norm.weight",
                       "layers.0.self_attn.k_proj.weight",
                       "layers.0.self_attn.o_proj.weight",
                       "layers.0.self_attn.q_norm.weight",
                       "layers.0.self_attn.q_proj.weight",
                       "layers.0.self_attn.v_proj.weight",
                       "norm.weight", "pre_fc_norm_embedding.weight",
                       "pre_fc_norm_hidden.weight"].map {
            GTurboIgnoredTensorV2(name: "mtp." + $0, reason: .unsupportedMTP)
        }
        return try GTurboManifestV2Codec.encode(.init(
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
            quantization: quantization, ignoredTensors: ignored,
            files: [
                "model_weights.bin": .init(size: 32_768, sha256: digest),
                "packed_experts/layout.json": .init(size: 1, sha256: digest),
            ],
            tensorRegions: [.init(
                name: "embed", file: "model_weights.bin", offset: 0,
                size: 16, shape: [1], storage: .affineInt4,
                quantizationCategory: .embedding)],
            expertsPerLayer: 256, numLayers: 40, expertStride: 16_384))
    }
}

private final class AwaitOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Never>?
    private var signaled = false

    func wait() async {
        await withCheckedContinuation { continuation in
            lock.lock()
            if signaled {
                lock.unlock()
                continuation.resume()
            } else {
                self.continuation = continuation
                lock.unlock()
            }
        }
    }

    func signal() {
        lock.lock()
        signaled = true
        let continuation = self.continuation
        self.continuation = nil
        lock.unlock()
        continuation?.resume()
    }
}
