import CryptoKit
import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

/// Opt-in P23 consumer for one explicitly authorised authentic image.
///
/// The independent reference is consumed as a completed, identity-bound JSON
/// artifact. Its preprocessing and prompt gates run before any Metal context
/// or text model is opened. A failed gate emits a bounded, machine-readable
/// line and throws, so a failed comparison remains useful evidence.
/// `TURBO_FIELDFARE_QWEN_VISION_PROMPT` is plain user text. The reference
/// prompt digest covers the complete string rendered by `QwenChatCodec` after
/// its one image part is inserted with the same options used for tokenization.
@Suite("Qwen real vision reference comparison", .serialized)
struct QwenRealVisionIntegrationTests {
    private static let vocabularySize = 248_320
    private static let eosTokenID: Int32 = 248_044
    private static let modelID = "Qwen/Qwen3.6-35B-A3B"
    private static let sourceRevision = "995ad96eacd98c81ed38be0c5b274b04031597b0"
    private static let absoluteToleranceDouble = 1e-5
    private static let relativeToleranceDouble = 1e-5
    private static let absoluteTolerance: Float = Float(absoluteToleranceDouble)
    private static let relativeTolerance: Float = Float(relativeToleranceDouble)
    private static let activationEnvironment = "TURBO_FIELDFARE_REAL_QWEN_VISION_PACK"
    private static let requiredEnvironmentKeys = [
        "TURBO_FIELDFARE_REAL_QWEN_ARTIFACT",
        "TURBO_FIELDFARE_REAL_QWEN_VISION_PACK",
        "TURBO_FIELDFARE_QWEN_VISION_REFERENCE_OUTPUT",
        "TURBO_FIELDFARE_QWEN_VISION_IMAGE",
        "TURBO_FIELDFARE_QWEN_VISION_PROMPT",
        "TURBO_FIELDFARE_QWEN_VISION_CASE_ID",
        "TURBO_FIELDFARE_QWEN_MANIFEST_SHA256",
        "TURBO_FIELDFARE_QWEN_VISION_MANIFEST_SHA256",
        "TURBO_FIELDFARE_QWEN_VISION_RECEIPT_SHA256",
        "TURBO_FIELDFARE_QWEN_POLICY_SHA256",
        "TURBO_FIELDFARE_QWEN_VISION_PROCESSOR_CONFIG_SHA256",
        "TURBO_FIELDFARE_QWEN_VISION_PAYLOAD_SHA256",
    ]

    @Test(.enabled(
        if: Self.isExplicitlyEnabled,
        "set TURBO_FIELDFARE_REAL_QWEN_VISION_PACK and the other explicit P23 inputs"))
    func authenticVisionMatchesPreparedReference() async throws {
        let configuration = try Configuration.fromEnvironment()
        let referenceData = try Data(contentsOf: configuration.referenceOutput)
        guard referenceData.count <= 32 * 1024 * 1024 else {
            throw Self.failure("reference", "reference output exceeds the 32 MiB contract bound")
        }
        let reference = try ReferenceDocument(
            data: referenceData,
            configuration: configuration)

        // Only metadata, manifest and the already-produced reference output
        // are consumed here. No model/Metal context is constructed until all
        // reference schema, identity and prompt metadata have been checked.
        let admission = try ModelFamilyGenerationSession.inspect(
            directoryURL: configuration.textDirectory)
        guard admission.family == .qwen3_6,
              let loadedIdentity = admission.verifiedIdentity else {
            throw Self.failure("identity", "text artifact was not admitted as Qwen 3.6")
        }
        try Self.validateTextIdentity(loadedIdentity, configuration: configuration,
                                      reference: reference)
        try Self.validateMetadataDigests(configuration)
        guard case .qwenV2(let manifest) = try ModelFamilyAdmission.classify(
            directoryURL: configuration.textDirectory),
              case .qwen3_6(let architecture) = manifest.architecture else {
            throw Self.failure("identity", "Qwen architecture was not present in the verified manifest")
        }
        let companion = try ModelFamilyGenerationSession.openVerifiedQwenVisionCompanion(
            directoryURL: configuration.textDirectory,
            loadedIdentity: loadedIdentity,
            visionPackURL: configuration.visionDirectory)
        try Self.validateVisionIdentity(
            reference, configuration: configuration,
            loadedIdentity: loadedIdentity,
            actualVisionManifest: companion.store.manifest)

        let context = try MetalContext()
        let preprocessor = QwenImagePreprocessor(device: context.device)
        let plan = try preprocessor.plan(fileURL: configuration.image)
        let pixels = try preprocessor.preprocess(plan)
        try Self.compareImageIdentity(
            plan: plan, pixels: pixels, reference: reference,
            configuration: configuration)

        let visionRuntime = try QwenVisionRuntime(
            context: context, store: companion.store)
        let visionFeatures = try await visionRuntime.process([pixels])
        guard visionFeatures.count == 1 else {
            throw Self.failure("vision", "production returned \(visionFeatures.count) images")
        }
        let feature = visionFeatures[0]
        let tokenizer = try QwenTokenizer.load(from: configuration.textDirectory)
        let codec = QwenChatCodec(tokenizer: tokenizer)
        let promptMessage = ModelChatMessage(
            role: .user,
            content: .parts([
                .text(configuration.prompt),
                .image(.init(id: "p23-image")),
            ]))
        let renderedPrompt = try codec.renderPrompt(
            messages: [promptMessage], tools: [], options: .init(enableThinking: false))
        let imageMarker = "<|vision_start|><|image_pad|><|vision_end|>"
        guard !configuration.prompt.contains(imageMarker),
              renderedPrompt.components(separatedBy: imageMarker).count == 2 else {
            throw Self.failure("prompt", "plain user text or rendered prompt has an invalid image marker count")
        }
        guard Data(renderedPrompt.utf8).count == reference.promptUTF8Bytes,
              Self.sha256(Data(renderedPrompt.utf8)) == reference.promptSHA256 else {
            throw Self.failure(
                "prompt",
                "plain user text plus the pinned image template differs from reference prompt metadata")
        }
        let encoded = try codec.encodePrompt(
            messages: [promptMessage], tools: [],
            options: .init(enableThinking: false))
        let normalized = try normalizeQwenCodecImageFrames(
            encoded, architecture: architecture)
        let rendered = try MultimodalPromptRenderer.expandingQwenImageTokens(
            normalized, features: [feature], architecture: architecture)
        try Self.comparePromptMetadata(rendered, reference: reference)
        try Self.compareVisionOutput(feature, reference: reference)

        // The exact patch, geometry, token, feature and M-RoPE gates have
        // passed. The full Qwen text model is opened once, and one runner is
        // reused for the reference's fixed one-or-two continuation steps.
        guard case .qwen(let model) = try ModelFamilyRuntime.load(
            directoryURL: configuration.textDirectory,
            device: context.device,
            integrityPolicy: .fullSha256) else {
            throw Self.failure("identity", "verified Qwen artifact did not load as a Qwen text model")
        }
        let prepared = try QwenPreparedPrefill(
            tokenIDs: rendered.effectiveTokenIDs,
            featureOverrides: rendered.imageSpans.map {
                QwenPreparedFeatureOverride(
                    tokenRange: $0.tokenRange, owner: $0.features.owner)
            },
            positions: rendered.positionPlan.positions,
            textRoPEDelta: rendered.positionPlan.textRoPEDelta)

        let runner = try model.makeRunner(context: context)
        var generated: [Int32] = []
        for (index, referenceStep) in reference.steps.enumerated() {
            let result: QwenTextHybridResult
            if index == 0 {
                result = try await runner.prefill(prepared: prepared)
            } else {
                guard let previous = generated.last else {
                    throw Self.failure("logits", "reference requested a decode step without a previous token")
                }
                result = try await runner.decode(token: previous)
            }
            let expectedInput = index == 0
                ? rendered.effectiveTokenIDs
                : [generated[index - 1]]
            guard referenceStep.inputTokenIDs == expectedInput else {
                throw Self.failure(
                    "tokens",
                    "reference step \(index) input IDs do not match the production request")
            }
            try Self.compareLogits(
                result.output.logits,
                reference: referenceStep.logits,
                expectedGreedyTokenID: referenceStep.greedyTokenID,
                step: index)
            guard result.output.state.sequenceLength
                    == referenceStep.sequenceLengthAfterForward else {
                throw Self.failure(
                    "sequence",
                    "step \(index) sequence length differs from the reference")
            }
            let greedy = Self.firstArgmax(result.output.logits)
            generated.append(greedy)
            if greedy == Self.eosTokenID { break }
        }
        guard generated == reference.generatedTokenIDs else {
            throw Self.failure("tokens", "generated token IDs differ from the reference")
        }
        let stopReason = generated.last == Self.eosTokenID ? "eos" : "maxTokens"
        guard stopReason == reference.stopReason else {
            throw Self.failure("stop", "stop reason \(stopReason) differs from \(reference.stopReason)")
        }
        guard tokenizer.decode(generated, skipSpecialTokens: false)
                == reference.decodedText else {
            throw Self.failure("text", "decoded continuation differs from the reference")
        }
    }

    private static var isExplicitlyEnabled: Bool {
        let environment = ProcessInfo.processInfo.environment
        return requiredEnvironmentKeys.allSatisfy { key in
            guard let value = environment[key] else { return false }
            return !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
    }

    private struct Configuration {
        let textDirectory: URL
        let visionDirectory: URL
        let referenceOutput: URL
        let image: URL
        let prompt: String
        let caseID: String
        let textManifestSHA256: String
        let visionManifestSHA256: String
        let visionReceiptSHA256: String
        let policySHA256: String
        let processorConfigSHA256: String
        let visionPayloadSHA256: String

        static func fromEnvironment() throws -> Self {
            let environment = ProcessInfo.processInfo.environment
            func required(_ name: String) throws -> String {
                guard let value = environment[name], !value.isEmpty else {
                    throw QwenRealVisionIntegrationTests.failure(
                        "configuration", "missing \(name)")
                }
                return value
            }
            func path(_ name: String, directory: Bool = false) throws -> URL {
                let value = try required(name)
                let requested = URL(fileURLWithPath: value).standardizedFileURL
                let url = requested.resolvingSymlinksInPath().standardizedFileURL
                guard requested.path.hasPrefix("/"), requested == url else {
                    throw QwenRealVisionIntegrationTests.failure(
                        "configuration", "\(name) must be an existing canonical physical path")
                }
                var isDirectory: ObjCBool = false
                let exists = FileManager.default.fileExists(
                    atPath: url.path, isDirectory: &isDirectory)
                guard exists, (directory ? isDirectory.boolValue : !isDirectory.boolValue) else {
                    throw QwenRealVisionIntegrationTests.failure(
                        "configuration", "\(name) is not an existing \(directory ? "directory" : "file")")
                }
                return url
            }
            let text = try path("TURBO_FIELDFARE_REAL_QWEN_ARTIFACT", directory: true)
            let vision = try path(activationEnvironment, directory: true)
            let reference = try path("TURBO_FIELDFARE_QWEN_VISION_REFERENCE_OUTPUT")
            let image = try path("TURBO_FIELDFARE_QWEN_VISION_IMAGE")
            guard text != vision, text != reference, text != image,
                  vision != reference, vision != image else {
                throw QwenRealVisionIntegrationTests.failure(
                    "configuration", "model, companion, output and image paths must be distinct")
            }
            func isWithin(_ child: URL, _ root: URL) -> Bool {
                let childComponents = child.standardizedFileURL.pathComponents
                let rootComponents = root.standardizedFileURL.pathComponents
                return childComponents.starts(with: rootComponents)
            }
            guard !isWithin(reference, text), !isWithin(reference, vision),
                  !isWithin(image, text), !isWithin(image, vision) else {
                throw QwenRealVisionIntegrationTests.failure(
                    "configuration", "reference output and image must be outside both model directories")
            }
            let prompt = try required("TURBO_FIELDFARE_QWEN_VISION_PROMPT")
            guard !prompt.isEmpty else {
                throw QwenRealVisionIntegrationTests.failure("configuration", "vision prompt is empty")
            }
            func digest(_ name: String) throws -> String {
                let value = try required(name)
                guard value.count == 64,
                      value.unicodeScalars.allSatisfy({
                          ($0.value >= 48 && $0.value <= 57)
                            || ($0.value >= 97 && $0.value <= 102)
                      }) else {
                    throw QwenRealVisionIntegrationTests.failure(
                        "configuration", "\(name) is not a lowercase SHA256")
                }
                return value
            }
            return Self(
                textDirectory: text, visionDirectory: vision,
                referenceOutput: reference, image: image,
                prompt: prompt,
                caseID: try required("TURBO_FIELDFARE_QWEN_VISION_CASE_ID"),
                textManifestSHA256: try digest("TURBO_FIELDFARE_QWEN_MANIFEST_SHA256"),
                visionManifestSHA256: try digest("TURBO_FIELDFARE_QWEN_VISION_MANIFEST_SHA256"),
                visionReceiptSHA256: try digest("TURBO_FIELDFARE_QWEN_VISION_RECEIPT_SHA256"),
                policySHA256: try digest("TURBO_FIELDFARE_QWEN_POLICY_SHA256"),
                processorConfigSHA256: try digest("TURBO_FIELDFARE_QWEN_VISION_PROCESSOR_CONFIG_SHA256"),
                visionPayloadSHA256: try digest("TURBO_FIELDFARE_QWEN_VISION_PAYLOAD_SHA256"))
        }
    }

    private struct ReferenceDocument {
        let textIdentity: [String: Any]
        let visionIdentity: [String: Any]
        let item: [String: Any]
        let steps: [ReferenceStep]
        let effectiveTokenIDs: [Int32]
        let mmTokenTypeIDs: [Int32]
        let imageGridTHW: [Int]
        let padRows: [Int]
        let processorBF16: TensorRecord
        let mergerOutput: TensorRecord
        let positionIDs: [Int32]
        let mropePositionDelta: Int
        let promptUTF8Bytes: Int
        let promptSHA256: String
        let maxNewTokens: Int
        let generatedTokenIDs: [Int32]
        let decodedText: String
        let stopReason: String

        init(data: Data, configuration: Configuration) throws {
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw QwenRealVisionIntegrationTests.failure("reference", "output is not a JSON object")
            }
            try QwenRealVisionIntegrationTests.requireKeys(root, ["schemaVersion", "status", "textIdentity",
                                         "visionIdentity", "referenceEnvironment", "reader",
                                         "request", "comparisonRules", "cases", "resources"],
                                  at: "reference")
            guard root["schemaVersion"] as? String == "qwen36-quantized-vision-reference-v1",
                  root["status"] as? String == "complete" else {
                throw QwenRealVisionIntegrationTests.failure("reference", "reference is not a complete P23 result")
            }
            guard let text = root["textIdentity"] as? [String: Any],
                  let vision = root["visionIdentity"] as? [String: Any],
                  let environment = root["referenceEnvironment"] as? [String: Any],
                  let reader = root["reader"] as? [String: Any],
                  let rules = root["comparisonRules"] as? [String: Any],
                  let request = root["request"] as? [String: Any],
                  let cases = root["cases"] as? [[String: Any]],
                  let resources = root["resources"] as? [String: Any] else {
                throw QwenRealVisionIntegrationTests.failure("reference", "reference identity/rules/cases have wrong types")
            }
            try Self.validateEnvelopeSchema(
                text: text, vision: vision, environment: environment,
                reader: reader, rules: rules, request: request,
                cases: cases, resources: resources)
            guard QwenRealVisionIntegrationTests.isDigest(QwenRealVisionIntegrationTests.string(request, "sha256")) else {
                throw QwenRealVisionIntegrationTests.failure(
                    "reference", "request record SHA256 is not a lowercase digest")
            }
            guard (request["caseCount"] as? NSNumber)?.intValue == cases.count,
                  cases.count == 1 else {
                throw QwenRealVisionIntegrationTests.failure(
                    "reference", "consumer requires exactly one explicitly authorised reference case")
            }
            guard let referenceCase = cases.first,
                  QwenRealVisionIntegrationTests.string(referenceCase, "id") == configuration.caseID else {
                throw QwenRealVisionIntegrationTests.failure("reference", "selected case ID is absent")
            }
            try Self.validateRules(rules)
            try QwenRealVisionIntegrationTests.requireKeys(referenceCase, ["id", "image", "prompt", "effectiveTokenIDs",
                                                  "mmTokenTypeIDs", "imageGridTHW", "padRows",
                                                  "processorFloat32", "processorBF16", "towerOutput",
                                                  "mergerOutput", "positionIDs", "mropePositionDelta",
                                                  "maxNewTokens", "generatedTokenIDs", "decodedText",
                                                  "stopReason", "steps"], at: "case")
            let stepsAny = referenceCase["steps"] as? [[String: Any]] ?? []
            guard (1...2).contains(stepsAny.count) else {
                throw QwenRealVisionIntegrationTests.failure("reference", "reference must contain one or two steps")
            }
            self.textIdentity = text
            self.visionIdentity = vision
            self.item = referenceCase
            effectiveTokenIDs = try QwenRealVisionIntegrationTests.int32Array(referenceCase["effectiveTokenIDs"], at: "effectiveTokenIDs")
            mmTokenTypeIDs = try QwenRealVisionIntegrationTests.int32Array(referenceCase["mmTokenTypeIDs"], at: "mmTokenTypeIDs")
            imageGridTHW = try QwenRealVisionIntegrationTests.intArray(referenceCase["imageGridTHW"], at: "imageGridTHW")
            padRows = try QwenRealVisionIntegrationTests.intArray(referenceCase["padRows"], at: "padRows")
            guard imageGridTHW.count == 3 else {
                throw QwenRealVisionIntegrationTests.failure("reference", "imageGridTHW is not [T,H,W]")
            }
            processorBF16 = try TensorRecord(value: referenceCase["processorBF16"], at: "processorBF16")
            let processorFloat32 = try TensorRecord(value: referenceCase["processorFloat32"], at: "processorFloat32")
            guard processorFloat32.shape == processorBF16.shape,
                  processorFloat32.dtype == "float32-le",
                  processorBF16.dtype == "bfloat16-le-rne" else {
                throw QwenRealVisionIntegrationTests.failure("reference", "processor tensor metadata differs")
            }
            _ = try TensorRecord(value: referenceCase["towerOutput"], at: "towerOutput")
            mergerOutput = try TensorRecord(value: referenceCase["mergerOutput"], at: "mergerOutput", requireBytes: true)
            let nestedPositions = try QwenRealVisionIntegrationTests.nestedInt32(referenceCase["positionIDs"], at: "positionIDs")
            guard nestedPositions.shape.count == 3,
                  nestedPositions.shape[0] == 3,
                  nestedPositions.shape[1] == 1 else {
                throw QwenRealVisionIntegrationTests.failure("reference", "positionIDs is not [3,1,sequence]")
            }
            positionIDs = nestedPositions.values
            mropePositionDelta = try QwenRealVisionIntegrationTests.intValue(referenceCase["mropePositionDelta"], at: "mropePositionDelta")
            let prompt = try Self.object(referenceCase["prompt"], at: "case.prompt")
            promptUTF8Bytes = try QwenRealVisionIntegrationTests.intValue(prompt["utf8Bytes"], at: "case.prompt.utf8Bytes")
            promptSHA256 = try QwenRealVisionIntegrationTests.stringValue(prompt["sha256"], at: "case.prompt.sha256")
            guard promptUTF8Bytes > 0, QwenRealVisionIntegrationTests.isDigest(promptSHA256) else {
                throw QwenRealVisionIntegrationTests.failure("reference", "invalid case prompt metadata")
            }
            maxNewTokens = try QwenRealVisionIntegrationTests.intValue(referenceCase["maxNewTokens"], at: "case.maxNewTokens")
            guard (1...2).contains(maxNewTokens) else {
                throw QwenRealVisionIntegrationTests.failure(
                    "reference", "maxNewTokens is outside the preregistered bound")
            }
            guard stepsAny.count <= maxNewTokens else {
                throw QwenRealVisionIntegrationTests.failure(
                    "reference", "reference contains more decode steps than maxNewTokens")
            }
            generatedTokenIDs = try QwenRealVisionIntegrationTests.int32Array(referenceCase["generatedTokenIDs"], at: "generatedTokenIDs")
            decodedText = try QwenRealVisionIntegrationTests.stringValue(referenceCase["decodedText"], at: "decodedText")
            stopReason = try QwenRealVisionIntegrationTests.stringValue(referenceCase["stopReason"], at: "stopReason")
            guard stopReason == "eos" || stopReason == "maxTokens",
                  generatedTokenIDs.count == stepsAny.count else {
                throw QwenRealVisionIntegrationTests.failure("reference", "invalid stop or generated-token count")
            }
            if stopReason == "eos" {
                guard let lastToken = generatedTokenIDs.last,
                      lastToken == QwenRealVisionIntegrationTests.eosTokenID else {
                    throw QwenRealVisionIntegrationTests.failure("reference", "eos stop does not end in EOS")
                }
            } else {
                guard generatedTokenIDs.count == maxNewTokens else {
                    throw QwenRealVisionIntegrationTests.failure("reference", "maxTokens stop has the wrong token count")
                }
            }
            steps = try stepsAny.enumerated().map {
                try ReferenceStep(value: $0.element, expectedIndex: $0.offset)
            }
        }

        private static func validateEnvelopeSchema(
            text: [String: Any],
            vision: [String: Any],
            environment: [String: Any],
            reader: [String: Any],
            rules: [String: Any],
            request: [String: Any],
            cases: [[String: Any]],
            resources: [String: Any]
        ) throws {
            try requireKeys(text, [
                "manifestSHA256", "quantizationPolicySHA256", "modelID",
                "sourceRevision", "sourceIndexSHA256", "configSHA256", "files",
            ], at: "textIdentity")
            try requireKeys(vision, [
                "manifestSHA256", "receiptSHA256", "artifactKind", "modelID",
                "sourceRevision", "compatibleTextManifestSHA256", "processorProfile",
                "processorConfigSHA256", "visionPayloadSHA256", "supportsStillImages",
                "supportsVideo", "files",
            ], at: "visionIdentity")
            let profile = try object(vision["processorProfile"], at: "visionIdentity.processorProfile")
            try requireKeys(profile, [
                "processorClass", "imageProcessorType", "patchSize",
                "temporalPatchSize", "spatialMergeSize",
            ], at: "visionIdentity.processorProfile")
            try validateFileRecords(text["files"], at: "textIdentity.files")
            try validateFileRecords(vision["files"], at: "visionIdentity.files")

            try requireKeys(environment, [
                "pythonVersion", "torchVersion", "transformersVersion", "numpyVersion",
                "transformersCommit", "transformersTree", "modelingSourceSHA256",
                "configurationSourceSHA256", "platform", "processor", "device", "dtype",
                "attentionImplementation", "expertsImplementation", "offline",
                "deterministicAlgorithms", "torchThreads", "torchInteropThreads",
                "pillowVersion", "processorClass", "visionModelClass", "processorMode",
                "executedTransformersSources",
            ], at: "referenceEnvironment")
            let offline = try object(environment["offline"], at: "referenceEnvironment.offline")
            try requireKeys(offline, [
                "PYTHONHASHSEED", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "USE_HUB_KERNELS",
            ], at: "referenceEnvironment.offline")
            guard let sources = environment["executedTransformersSources"] as? [[String: Any]] else {
                throw failure("reference", "referenceEnvironment.executedTransformersSources is not an array")
            }
            let sourcePaths = sources.map {
                QwenRealVisionIntegrationTests.string($0, "path")
            }
            guard !sourcePaths.isEmpty,
                  sourcePaths == sourcePaths.sorted(),
                  Set(sourcePaths).count == sourcePaths.count else {
                throw failure("reference", "referenceEnvironment source records are not sorted and unique")
            }
            for (index, source) in sources.enumerated() {
                try requireKeys(source, ["path", "sha256"],
                                at: "referenceEnvironment.executedTransformersSources[\(index)]")
                guard !QwenRealVisionIntegrationTests.string(source, "path").isEmpty,
                      QwenRealVisionIntegrationTests.isDigest(QwenRealVisionIntegrationTests.string(source, "sha256")) else {
                    throw failure("reference", "referenceEnvironment source record is invalid")
                }
            }
            guard QwenRealVisionIntegrationTests.string(environment, "pythonVersion") == "3.12.3",
                  QwenRealVisionIntegrationTests.string(environment, "torchVersion") == "2.10.0",
                  QwenRealVisionIntegrationTests.string(environment, "transformersVersion") == "5.18.0.dev0",
                  QwenRealVisionIntegrationTests.string(environment, "numpyVersion") == "2.4.3",
                  QwenRealVisionIntegrationTests.string(environment, "transformersCommit") ==
                    "bd15bc95a89e728bbc1224084eb3b5829428c353",
                  QwenRealVisionIntegrationTests.string(environment, "transformersTree") ==
                    "80eb369e589827bc7ed45b3a1f0ead5457097535",
                  QwenRealVisionIntegrationTests.string(environment, "platform") == "Darwin",
                  QwenRealVisionIntegrationTests.string(environment, "device") == "cpu",
                  QwenRealVisionIntegrationTests.string(environment, "dtype") == "float32",
                  QwenRealVisionIntegrationTests.string(environment, "attentionImplementation") == "eager",
                  QwenRealVisionIntegrationTests.string(environment, "expertsImplementation") == "eager",
                  QwenRealVisionIntegrationTests.string(environment, "processorClass") == "Qwen3VLProcessor",
                  QwenRealVisionIntegrationTests.string(environment, "visionModelClass") == "Qwen3_5MoeVisionModel",
                  QwenRealVisionIntegrationTests.string(environment, "processorMode") == "pinned-local-only",
                  environment["deterministicAlgorithms"] as? Bool == true,
                  QwenRealVisionIntegrationTests.int(environment, "torchInteropThreads") == 1,
                  (1...12).contains(QwenRealVisionIntegrationTests.int(environment, "torchThreads")),
                  QwenRealVisionIntegrationTests.isDigest(QwenRealVisionIntegrationTests.string(environment, "modelingSourceSHA256")),
                  QwenRealVisionIntegrationTests.isDigest(QwenRealVisionIntegrationTests.string(environment, "configurationSourceSHA256")),
                  QwenRealVisionIntegrationTests.string(offline, "PYTHONHASHSEED") == "0",
                  QwenRealVisionIntegrationTests.string(offline, "HF_HUB_OFFLINE") == "1",
                  QwenRealVisionIntegrationTests.string(offline, "TRANSFORMERS_OFFLINE") == "1",
                  QwenRealVisionIntegrationTests.string(offline, "USE_HUB_KERNELS") == "0",
                  QwenRealVisionIntegrationTests.string(environment, "pillowVersion") == "10.3.0" else {
                throw failure("reference", "reference environment is not the pinned offline CPU environment")
            }

            try requireKeys(reader, [
                "schema", "scriptSHA256", "textReferenceScriptSHA256",
                "maximumTransientDecodeBytes", "maximumLiveDecodedTextLayerBytes",
                "runtimeModulesImported", "bf16CheckpointRead",
            ], at: "reader")
            guard QwenRealVisionIntegrationTests.string(reader, "schema") == "gturbo-v2-qwen-vision-independent-v1",
                  QwenRealVisionIntegrationTests.isDigest(QwenRealVisionIntegrationTests.string(reader, "scriptSHA256")),
                  QwenRealVisionIntegrationTests.string(reader, "textReferenceScriptSHA256") ==
                    "9b2463fd8935a17c049b3399f9ced52b62c554defaf3038894f6c21e4815a50d",
                  reader["runtimeModulesImported"] as? Bool == false,
                  reader["bf16CheckpointRead"] as? Bool == false,
                  QwenRealVisionIntegrationTests.int(reader, "maximumTransientDecodeBytes") == 64 * 1024 * 1024,
                  QwenRealVisionIntegrationTests.int(reader, "maximumLiveDecodedTextLayerBytes") > 0 else {
                throw failure("reference", "reader provenance is not the pinned independent reader")
            }
            try requireKeys(request, ["sha256", "caseCount"], at: "request")
            guard QwenRealVisionIntegrationTests.isDigest(QwenRealVisionIntegrationTests.string(request, "sha256")),
                  (1...2).contains(QwenRealVisionIntegrationTests.int(request, "caseCount")) else {
                throw failure("reference", "request metadata is outside the preregistered bound")
            }
            try requireKeys(rules, [
                "referenceAppliesPassFail", "preprocessingAdmission", "exact",
                "visionFeatures", "logits", "policy",
            ], at: "comparisonRules")
            let preprocessing = try object(
                rules["preprocessingAdmission"], at: "comparisonRules.preprocessingAdmission")
            try requireKeys(preprocessing, ["rule", "hardGateBeforeFeaturesOrLogits"],
                            at: "comparisonRules.preprocessingAdmission")
            let features = try object(rules["visionFeatures"], at: "comparisonRules.visionFeatures")
            try requireKeys(features, ["rule", "absolute", "relative", "predicate", "provenance"],
                            at: "comparisonRules.visionFeatures")
            let logits = try object(rules["logits"], at: "comparisonRules.logits")
            try requireKeys(logits, [
                "rule", "absolute", "relative", "predicate",
                "requiresExactFirstArgmax", "provenance",
            ], at: "comparisonRules.logits")
            try requireKeys(resources, [
                "decodedFloat32VisionWeightBytes", "maximumRetainedOfficialVisionTensorBytes",
                "maximumLiveDecodedTextLayerBytes", "maximumObservedPatchRows",
                "maximumObservedMergedRows", "finalJSONBytes",
            ], at: "resources")
            guard QwenRealVisionIntegrationTests.int(resources, "decodedFloat32VisionWeightBytes") == 1_786_284_992,
                  QwenRealVisionIntegrationTests.int(resources, "maximumRetainedOfficialVisionTensorBytes") > 0,
                  QwenRealVisionIntegrationTests.int(resources, "maximumLiveDecodedTextLayerBytes") > 0,
                  QwenRealVisionIntegrationTests.int(resources, "maximumObservedPatchRows") > 0,
                  QwenRealVisionIntegrationTests.int(resources, "maximumObservedMergedRows") > 0,
                  QwenRealVisionIntegrationTests.int(resources, "finalJSONBytes") > 0 else {
                throw failure("reference", "reference resource measurements are invalid")
            }

            let caseKeys = [
                "id", "image", "prompt", "effectiveTokenIDs", "mmTokenTypeIDs",
                "imageGridTHW", "padRows", "processorFloat32", "processorBF16",
                "towerOutput", "mergerOutput", "positionIDs", "mropePositionDelta",
                "maxNewTokens", "generatedTokenIDs", "decodedText", "stopReason", "steps",
            ]
            let imageKeys = [
                "basename", "bytes", "width", "height", "format", "mode", "frameCount",
                "orientation", "orientedWidth", "orientedHeight", "targetWidth", "targetHeight",
                "sha256",
            ]
            for (index, item) in cases.enumerated() {
                try requireKeys(item, caseKeys, at: "cases[\(index)]")
                let image = try object(item["image"], at: "cases[\(index)].image")
                try requireKeys(image, imageKeys, at: "cases[\(index)].image")
                guard !QwenRealVisionIntegrationTests.string(image, "basename").isEmpty,
                      !QwenRealVisionIntegrationTests.string(image, "format").isEmpty,
                      !QwenRealVisionIntegrationTests.string(image, "mode").isEmpty,
                      QwenRealVisionIntegrationTests.isDigest(QwenRealVisionIntegrationTests.string(image, "sha256")),
                      QwenRealVisionIntegrationTests.int(image, "bytes") > 0,
                      QwenRealVisionIntegrationTests.int(image, "width") > 0,
                      QwenRealVisionIntegrationTests.int(image, "height") > 0,
                      QwenRealVisionIntegrationTests.int(image, "orientedWidth") > 0,
                      QwenRealVisionIntegrationTests.int(image, "orientedHeight") > 0,
                      QwenRealVisionIntegrationTests.int(image, "targetWidth") > 0,
                      QwenRealVisionIntegrationTests.int(image, "targetHeight") > 0,
                      QwenRealVisionIntegrationTests.int(image, "frameCount") == 1,
                      (1...8).contains(QwenRealVisionIntegrationTests.int(image, "orientation")) else {
                    throw failure("reference", "cases[\(index)].image has invalid raw or geometry metadata")
                }
                let prompt = try object(item["prompt"], at: "cases[\(index)].prompt")
                try requireKeys(prompt, ["utf8Bytes", "sha256"], at: "cases[\(index)].prompt")
                guard (1...64 * 1024).contains(QwenRealVisionIntegrationTests.int(prompt, "utf8Bytes")),
                      QwenRealVisionIntegrationTests.isDigest(QwenRealVisionIntegrationTests.string(prompt, "sha256")) else {
                    throw failure("reference", "cases[\(index)].prompt metadata is invalid")
                }
                for tensorName in ["processorFloat32", "processorBF16", "towerOutput"] {
                    let tensor = try object(item[tensorName], at: "cases[\(index)].\(tensorName)")
                    try requireKeys(tensor, ["shape", "dtype", "byteCount", "sha256"],
                                    at: "cases[\(index)].\(tensorName)")
                }
                let merger = try object(item["mergerOutput"], at: "cases[\(index)].mergerOutput")
                try requireKeys(merger, ["shape", "dtype", "byteCount", "sha256", "encoding", "bytes"],
                                at: "cases[\(index)].mergerOutput")
                guard let steps = item["steps"] as? [[String: Any]] else {
                    throw failure("reference", "cases[\(index)].steps is not an array")
                }
                for (stepIndex, step) in steps.enumerated() {
                    try requireKeys(step, [
                        "step", "inputTokenIDs", "sequenceLengthAfterForward", "logits",
                        "greedyTokenID", "topK",
                    ], at: "cases[\(index)].steps[\(stepIndex)]")
                    let logits = try object(
                        step["logits"], at: "cases[\(index)].steps[\(stepIndex)].logits")
                    try requireKeys(logits, ["dtype", "encoding", "count", "sha256", "bytes"],
                                    at: "cases[\(index)].steps[\(stepIndex)].logits")
                    guard let topK = step["topK"] as? [[String: Any]], topK.count == 5 else {
                        throw failure("reference", "cases[\(index)].steps[\(stepIndex)].topK is not five entries")
                    }
                    for (topIndex, entry) in topK.enumerated() {
                        try requireKeys(entry, ["tokenID", "logit"],
                                        at: "cases[\(index)].steps[\(stepIndex)].topK[\(topIndex)]")
                    }
                }
            }
        }

        private static func object(_ value: Any?, at path: String) throws -> [String: Any] {
            guard let value = value as? [String: Any] else {
                throw failure("reference", "\(path) is not an object")
            }
            return value
        }

        private static func validateFileRecords(_ value: Any?, at path: String) throws {
            guard let records = value as? [[String: Any]], !records.isEmpty else {
                throw failure("reference", "\(path) is not a nonempty file-record array")
            }
            let paths = records.map { QwenRealVisionIntegrationTests.string($0, "path") }
            guard paths == paths.sorted(), Set(paths).count == paths.count else {
                throw failure("reference", "\(path) is not sorted and unique")
            }
            for (index, record) in records.enumerated() {
                try requireKeys(record, ["path", "size", "sha256"], at: "\(path)[\(index)]")
                guard !QwenRealVisionIntegrationTests.string(record, "path").isEmpty,
                      QwenRealVisionIntegrationTests.int(record, "size") >= 0,
                      QwenRealVisionIntegrationTests.isDigest(
                        QwenRealVisionIntegrationTests.string(record, "sha256")) else {
                    throw failure("reference", "\(path)[\(index)] contains invalid file metadata")
                }
            }
        }

        private static func validateRules(_ rules: [String: Any]) throws {
            guard rules["referenceAppliesPassFail"] as? Bool == false,
                  let preprocessing = rules["preprocessingAdmission"] as? [String: Any],
                  preprocessing["rule"] as? String == "exact-bf16-patch-shape-byte-count-and-sha256",
                  preprocessing["hardGateBeforeFeaturesOrLogits"] as? Bool == true,
                  let features = rules["visionFeatures"] as? [String: Any],
                  features["rule"] as? String == "elementwise",
                  QwenRealVisionIntegrationTests.double(features["absolute"]) == absoluteToleranceDouble,
                  QwenRealVisionIntegrationTests.double(features["relative"]) == relativeToleranceDouble,
                  let logits = rules["logits"] as? [String: Any],
                  logits["rule"] as? String == "vector-global",
                  QwenRealVisionIntegrationTests.double(logits["absolute"]) == absoluteToleranceDouble,
                  QwenRealVisionIntegrationTests.double(logits["relative"]) == relativeToleranceDouble,
                  logits["predicate"] as? String
                    == "maxAbs<=absolute+relative*max(maxAbsCandidate,maxAbsReference)",
                  logits["provenance"] as? String == "P3/P22",
                  logits["requiresExactFirstArgmax"] as? Bool == true,
                  features["predicate"] as? String
                    == "abs(candidate-reference)<=absolute+relative*abs(reference)",
                  features["provenance"] as? String == "P16",
                  rules["policy"] as? String
                    == "preregistered-authentic-cap; no-post-result-widening",
                  rules["exact"] as? [String]
                    == ["rawImageIdentity", "orientedAndTargetGeometry", "imageGridTHW",
                        "effectiveTokenIDs", "mmTokenTypeIDs", "padRows", "positionIDs",
                        "mropePositionDelta", "sequenceLength", "greedyTokenID", "stopReason"] else {
                throw QwenRealVisionIntegrationTests.failure(
                    "reference", "comparison rules do not match the preregistered gates")
            }
        }
    }

    private struct ReferenceStep {
        let inputTokenIDs: [Int32]
        let sequenceLengthAfterForward: Int
        let logits: LogitRecord
        let greedyTokenID: Int32

        init(value: [String: Any], expectedIndex: Int) throws {
            try QwenRealVisionIntegrationTests.requireKeys(
                value, ["step", "inputTokenIDs", "sequenceLengthAfterForward", "logits", "greedyTokenID", "topK"], at: "step")
            guard QwenRealVisionIntegrationTests.int(value, "step") == expectedIndex,
                  let topK = value["topK"] as? [[String: Any]], topK.count == 5 else {
                throw QwenRealVisionIntegrationTests.failure("reference", "step index or topK shape is invalid")
            }
            for (index, entry) in topK.enumerated() {
                try QwenRealVisionIntegrationTests.requireKeys(
                    entry, ["tokenID", "logit"], at: "step.topK[\(index)]")
                let tokenID = QwenRealVisionIntegrationTests.int(entry, "tokenID")
                guard (0..<QwenRealVisionIntegrationTests.vocabularySize).contains(tokenID),
                      let logit = (entry["logit"] as? NSNumber)?.doubleValue,
                      logit.isFinite else {
                    throw QwenRealVisionIntegrationTests.failure("reference", "step topK entry is invalid")
                }
            }
            inputTokenIDs = try QwenRealVisionIntegrationTests.int32Array(value["inputTokenIDs"], at: "step.inputTokenIDs")
            sequenceLengthAfterForward = try QwenRealVisionIntegrationTests.intValue(value["sequenceLengthAfterForward"], at: "step.sequenceLengthAfterForward")
            logits = try LogitRecord(value: value["logits"], at: "step.logits")
            greedyTokenID = try QwenRealVisionIntegrationTests.int32Value(value["greedyTokenID"], at: "step.greedyTokenID")
            guard sequenceLengthAfterForward > 0,
                  greedyTokenID >= 0,
                  greedyTokenID < Int32(QwenRealVisionIntegrationTests.vocabularySize) else {
                throw QwenRealVisionIntegrationTests.failure("reference", "step metadata is outside model bounds")
            }
        }
    }

    private struct LogitRecord {
        let dtype: String
        let encoding: String
        let count: Int
        let sha256: String
        let bytes: Data

        init(value: Any?, at path: String) throws {
            guard let object = value as? [String: Any],
                  let dtype = object["dtype"] as? String,
                  let encoding = object["encoding"] as? String,
                  let count = (object["count"] as? NSNumber)?.intValue,
                  let sha256 = object["sha256"] as? String,
                  let encoded = object["bytes"] as? String else {
                throw QwenRealVisionIntegrationTests.failure("reference", "\(path) logit metadata is incomplete")
            }
            try QwenRealVisionIntegrationTests.requireKeys(
                object, ["dtype", "encoding", "count", "sha256", "bytes"], at: path)
            guard dtype == "float32-le",
                  encoding == "base64",
                  count == QwenRealVisionIntegrationTests.vocabularySize,
                  let bytes = Data(base64Encoded: encoded),
                  bytes.count == count * MemoryLayout<Float>.stride,
                  QwenRealVisionIntegrationTests.isDigest(sha256),
                  QwenRealVisionIntegrationTests.sha256(bytes) == sha256 else {
                throw QwenRealVisionIntegrationTests.failure(
                    "reference", "\(path) is not a complete finite-vocabulary logit record")
            }
            self.dtype = dtype
            self.encoding = encoding
            self.count = count
            self.sha256 = sha256
            self.bytes = bytes
        }
    }

    private struct TensorRecord {
        let shape: [Int]
        let dtype: String
        let byteCount: Int
        let sha256: String
        let bytes: Data?

        init(value: Any?, at path: String, requireBytes: Bool = false) throws {
            guard let object = value as? [String: Any],
                  let shapeValue = object["shape"],
                  let dtype = object["dtype"] as? String,
                  let byteCount = (object["byteCount"] as? NSNumber)?.intValue,
                  let sha256 = object["sha256"] as? String else {
                throw QwenRealVisionIntegrationTests.failure("reference", "\(path) tensor metadata is incomplete")
            }
            shape = try QwenRealVisionIntegrationTests.intArray(shapeValue, at: "\(path).shape")
            self.dtype = dtype
            self.byteCount = byteCount
            self.sha256 = sha256
            if let encoded = object["bytes"] as? String {
                guard (!requireBytes || object["encoding"] as? String == "base64"),
                      let data = Data(base64Encoded: encoded), data.count == byteCount,
                      QwenRealVisionIntegrationTests.sha256(data) == sha256 else {
                    throw QwenRealVisionIntegrationTests.failure("reference", "\(path) bytes do not match its digest")
                }
                bytes = data
            } else {
                guard !requireBytes else {
                    throw QwenRealVisionIntegrationTests.failure("reference", "\(path) omits required complete bytes")
                }
                bytes = nil
            }
            let scalarBytes: Int = switch dtype {
            case "float32-le": 4
            case "bfloat16-le-rne": 2
            default: 0
            }
            var elementCount = 1
            for dimension in shape {
                let (next, overflow) = elementCount.multipliedReportingOverflow(by: dimension)
                guard !overflow else {
                    throw QwenRealVisionIntegrationTests.failure("reference", "\(path) shape overflows")
                }
                elementCount = next
            }
            guard byteCount >= 0,
                  QwenRealVisionIntegrationTests.isDigest(sha256),
                  !shape.isEmpty, shape.allSatisfy({ $0 > 0 }),
                  scalarBytes > 0,
                  elementCount == byteCount / scalarBytes,
                  byteCount.isMultiple(of: scalarBytes) else {
                throw QwenRealVisionIntegrationTests.failure("reference", "\(path) has an invalid shape or byte count")
            }
        }
    }

    private struct NestedInt32 {
        let shape: [Int]
        let values: [Int32]
    }

    private enum ParityError: Error, CustomStringConvertible {
        case failure(gate: String, detail: String)

        var description: String {
            switch self {
            case .failure(let gate, let detail):
                return "P23 real vision \(gate) failure: \(detail)"
            }
        }
    }

    private static func validateTextIdentity(
        _ loaded: LoadedRuntimeIdentity,
        configuration: Configuration,
        reference: ReferenceDocument
    ) throws {
        guard loaded.modelID == modelID,
              loaded.sourceRevision == sourceRevision,
              loaded.textManifestSHA256 == configuration.textManifestSHA256,
              loaded.quantizationPolicySHA256 == configuration.policySHA256,
              string(reference.textIdentity, "manifestSHA256") == configuration.textManifestSHA256,
              string(reference.textIdentity, "quantizationPolicySHA256") == configuration.policySHA256,
              string(reference.textIdentity, "modelID") == modelID,
              string(reference.textIdentity, "sourceRevision") == sourceRevision,
              string(reference.textIdentity, "sourceIndexSHA256") == loaded.sourceIndexSHA256 else {
            throw failure("identity", "reference and loaded text identities differ")
        }
        let configURL = configuration.textDirectory.appendingPathComponent("config.json")
        guard let configData = try? Data(contentsOf: configURL),
              sha256(configData) == string(reference.textIdentity, "configSHA256") else {
            throw failure("identity", "reference config digest differs from the admitted text artifact")
        }
    }

    private static func validateMetadataDigests(_ configuration: Configuration) throws {
        let metadata: [(String, URL, String)] = [
            ("text manifest", configuration.textDirectory.appendingPathComponent("manifest.json"),
             configuration.textManifestSHA256),
            ("vision manifest", configuration.visionDirectory.appendingPathComponent("manifest.json"),
             configuration.visionManifestSHA256),
            ("vision receipt", configuration.visionDirectory.appendingPathComponent("verified-install.json"),
             configuration.visionReceiptSHA256),
            ("vision processor config",
             configuration.visionDirectory.appendingPathComponent("preprocessor_config.json"),
             configuration.processorConfigSHA256),
        ]
        for (name, url, expected) in metadata {
            guard let data = try? Data(contentsOf: url), sha256(data) == expected else {
                throw failure("identity", "\(name) bytes do not match the configured SHA256")
            }
        }
    }

    private static func validateVisionIdentity(
        _ reference: ReferenceDocument,
        configuration: Configuration,
        loadedIdentity: LoadedRuntimeIdentity,
        actualVisionManifest: GTurboVisionManifestV2
    ) throws {
        guard string(reference.visionIdentity, "manifestSHA256") == configuration.visionManifestSHA256,
              string(reference.visionIdentity, "receiptSHA256") == configuration.visionReceiptSHA256,
              string(reference.visionIdentity, "processorConfigSHA256") == configuration.processorConfigSHA256,
              string(reference.visionIdentity, "visionPayloadSHA256") == configuration.visionPayloadSHA256,
              string(reference.visionIdentity, "compatibleTextManifestSHA256") == loadedIdentity.textManifestSHA256,
              string(reference.visionIdentity, "artifactKind") == "qwen3_6_vision_companion",
              reference.visionIdentity["supportsStillImages"] as? Bool == true,
              reference.visionIdentity["supportsVideo"] as? Bool == false,
              string(reference.visionIdentity, "modelID") == modelID,
              string(reference.visionIdentity, "sourceRevision") == sourceRevision else {
            throw failure("identity", "reference and configured vision identities differ")
        }
        let profile = try object(
            reference.visionIdentity["processorProfile"],
            at: "visionIdentity.processorProfile")
        let actualProfile = actualVisionManifest.processorProfile
        guard actualVisionManifest.artifactKind == "qwen3_6_vision_companion",
              actualVisionManifest.modelID == modelID,
              actualVisionManifest.sourceRevision == sourceRevision,
              actualVisionManifest.processorConfigSHA256 == configuration.processorConfigSHA256,
              actualVisionManifest.visionPayloadSHA256 == configuration.visionPayloadSHA256,
              actualVisionManifest.supportsStillImages,
              !actualVisionManifest.supportsVideo,
              string(profile, "processorClass") == actualProfile.processorClass,
              string(profile, "imageProcessorType") == actualProfile.imageProcessorType,
              int(profile, "patchSize") == actualProfile.patchSize,
              int(profile, "temporalPatchSize") == actualProfile.temporalPatchSize,
              int(profile, "spatialMergeSize") == actualProfile.spatialMergeSize else {
            throw failure("identity", "reference processor profile differs from the verified vision manifest")
        }
        if case .verified(let vision) = loadedIdentity.vision {
            guard vision.compatibleTextManifestSHA256 == loadedIdentity.textManifestSHA256,
                  vision.processorConfigSHA256 == configuration.processorConfigSHA256,
                  vision.visionPayloadSHA256 == configuration.visionPayloadSHA256,
                  vision.sourceRevision == loadedIdentity.sourceRevision,
                  vision.supportsStillImages,
                  !vision.supportsVideo else {
                throw failure("identity", "prebound text identity does not expose this compatible still-image companion")
            }
        }
    }

    private static func compareImageIdentity(
        plan: QwenImagePlan,
        pixels: QwenVisionPixelBuffer,
        reference: ReferenceDocument,
        configuration: Configuration
    ) throws {
        guard let image = reference.item["image"] as? [String: Any],
              string(image, "basename") == configuration.image.lastPathComponent,
              int(image, "bytes") == plan.metadata.encodedBytes,
              int(image, "width") == plan.metadata.encodedWidth,
              int(image, "height") == plan.metadata.encodedHeight,
              int(image, "orientation") == plan.metadata.orientation,
              int(image, "orientedWidth") == plan.metadata.orientedWidth,
              int(image, "orientedHeight") == plan.metadata.orientedHeight,
              int(image, "targetWidth") == plan.geometry.processedWidth,
              int(image, "targetHeight") == plan.geometry.processedHeight,
              int(image, "frameCount") == 1,
              !string(image, "format").isEmpty,
              !string(image, "mode").isEmpty,
              string(image, "sha256") == pixels.imageDigest else {
            throw failure("preprocessing", "encoded image identity or geometry differs")
        }
        guard reference.imageGridTHW == [plan.geometry.gridT, plan.geometry.gridH, plan.geometry.gridW],
              plan.geometry.gridH * QwenVisionConfig.official.patchSize == plan.geometry.processedHeight,
              plan.geometry.gridW * QwenVisionConfig.official.patchSize == plan.geometry.processedWidth else {
            throw failure("preprocessing", "oriented or target geometry differs")
        }
        let patchByteCount = plan.geometry.patchRows
            * QwenVisionConfig.official.patchWidth
            * MemoryLayout<UInt16>.stride
        guard reference.processorBF16.shape == [
            plan.geometry.patchRows, QwenVisionConfig.official.patchWidth,
        ], reference.processorBF16.dtype == "bfloat16-le-rne",
              reference.processorBF16.byteCount == patchByteCount else {
            throw failure("preprocessing", "BF16 patch shape differs before numerical comparison")
        }
        guard patchByteCount == pixels.patchesBF16.length else {
            throw failure("preprocessing", "production patch byte count differs")
        }
        let patchBytes = Data(bytes: pixels.patchesBF16.contents(), count: patchByteCount)
        guard sha256(patchBytes) == reference.processorBF16.sha256,
              reference.processorBF16.byteCount == patchBytes.count else {
            throw failure("preprocessing", "BF16 patch bytes or SHA256 differ")
        }
    }

    private static func compareVisionOutput(
        _ feature: QwenVisionFeatures,
        reference: ReferenceDocument
    ) throws {
        let expectedGrid = try QwenVisionGrid(
            temporal: reference.imageGridTHW[0],
            height: reference.imageGridTHW[1],
            width: reference.imageGridTHW[2])
        guard feature.tokenCount == reference.imageGridTHW[0]
                * reference.imageGridTHW[1] * reference.imageGridTHW[2] / 4,
              feature.hiddenSize == QwenVisionConfig.official.outputHiddenSize,
              feature.grid == expectedGrid else {
            throw failure("vision", "production merger shape or grid differs")
        }
        let candidate = feature.owner.features()
        guard candidate.allSatisfy(\.isFinite),
              let referenceBytes = reference.mergerOutput.bytes else {
            throw failure("vision", "merger output is not finite or reference bytes are absent")
        }
        let expected = decodeFloat32(referenceBytes)
        guard reference.mergerOutput.shape == [feature.tokenCount, feature.hiddenSize],
              reference.mergerOutput.dtype == "float32-le",
              reference.mergerOutput.byteCount == expected.count * MemoryLayout<Float>.stride,
              expected.count == candidate.count else {
            throw failure("vision", "merger output shape differs")
        }
        for (index, pair) in zip(candidate, expected).enumerated() {
            let limit = absoluteTolerance + relativeTolerance * abs(pair.1)
            guard abs(pair.0 - pair.1) <= limit else {
                throw failure("vision", "merger element \(index) exceeds 1e-5 absolute/relative tolerance")
            }
        }
    }

    private static func comparePromptMetadata(
        _ rendered: QwenMultimodalPrefillInput,
        reference: ReferenceDocument
    ) throws {
        guard rendered.effectiveTokenIDs == reference.effectiveTokenIDs else {
            throw failure("tokens", "effectiveTokenIDs differ before model execution")
        }
        var types = [Int32](repeating: 0, count: rendered.effectiveTokenIDs.count)
        for span in rendered.imageSpans {
            for index in span.tokenRange { types[index] = 1 }
        }
        guard types == reference.mmTokenTypeIDs else {
            throw failure("tokens", "mmTokenTypeIDs differ before model execution")
        }
        guard let span = rendered.imageSpans.first,
              rendered.imageSpans.count == 1,
              Array(span.tokenRange) == reference.padRows else {
            throw failure("tokens", "image pad rows differ before model execution")
        }
        let positions = rendered.positionPlan.flattenedInt32x3
        let sequence = rendered.positionPlan.positions.count
        guard reference.positionIDs.count == sequence * 3 else {
            throw failure("positions", "reference M-RoPE position shape differs before model execution")
        }
        var axisMajorReference: [Int32] = []
        axisMajorReference.reserveCapacity(reference.positionIDs.count)
        for token in 0..<sequence {
            axisMajorReference.append(reference.positionIDs[token])
            axisMajorReference.append(reference.positionIDs[sequence + token])
            axisMajorReference.append(reference.positionIDs[2 * sequence + token])
        }
        guard positions == axisMajorReference,
              rendered.positionPlan.textRoPEDelta == reference.mropePositionDelta else {
            throw failure("positions", "M-RoPE position IDs or delta differ before model execution")
        }
    }

    private static func compareLogits(
        _ candidate: [Float],
        reference: LogitRecord,
        expectedGreedyTokenID: Int32,
        step: Int
    ) throws {
        guard candidate.count == vocabularySize,
              candidate.allSatisfy(\.isFinite) else {
            throw failure("logits", "step \(step) does not contain a finite complete vocabulary vector")
        }
        let expected = decodeFloat32(reference.bytes)
        guard expected.count == candidate.count, expected.allSatisfy(\.isFinite) else {
            throw failure("logits", "step \(step) reference vector is incomplete or non-finite")
        }
        let candidateMax = candidate.map(abs).max() ?? 0
        let referenceMax = expected.map(abs).max() ?? 0
        let maxDifference = zip(candidate, expected)
            .map { abs($0.0 - $0.1) }.max() ?? 0
        let limit = absoluteTolerance + relativeTolerance
            * max(candidateMax, referenceMax)
        guard maxDifference <= limit else {
            throw failure("logits", "step \(step) vector-global maxAbs exceeds preregistered tolerance")
        }
        let firstArgmax = firstArgmax(candidate)
        guard firstArgmax == expectedGreedyTokenID else {
            throw failure("logits", "step \(step) first argmax differs from the reference")
        }
    }

    private static func firstArgmax(_ values: [Float]) -> Int32 {
        var index = 0
        var best = -Float.infinity
        for (candidateIndex, value) in values.enumerated() where value > best {
            best = value
            index = candidateIndex
        }
        return Int32(index)
    }

    private static func requireKeys(
        _ object: [String: Any], _ keys: [String], at path: String
    ) throws {
        guard Set(object.keys) == Set(keys) else {
            throw failure("reference", "\(path) has unexpected keys")
        }
    }

    private static func object(_ value: Any?, at path: String) throws -> [String: Any] {
        guard let value = value as? [String: Any] else {
            throw failure("reference", "\(path) is not an object")
        }
        return value
    }

    private static func string(_ object: [String: Any], _ key: String) -> String {
        object[key] as? String ?? ""
    }

    private static func int(_ object: [String: Any], _ key: String) -> Int {
        (object[key] as? NSNumber)?.intValue ?? Int.min
    }

    private static func stringValue(_ value: Any?, at path: String) throws -> String {
        guard let value = value as? String else {
            throw failure("reference", "\(path) is not a string")
        }
        return value
    }

    private static func intValue(_ value: Any?, at path: String) throws -> Int {
        guard let value = value as? NSNumber else {
            throw failure("reference", "\(path) is not an integer")
        }
        return value.intValue
    }

    private static func int32Value(_ value: Any?, at path: String) throws -> Int32 {
        let value = try intValue(value, at: path)
        guard let result = Int32(exactly: value) else {
            throw failure("reference", "\(path) is outside Int32")
        }
        return result
    }

    private static func intArray(_ value: Any?, at path: String) throws -> [Int] {
        guard let values = value as? [Any] else {
            throw failure("reference", "\(path) is not an array")
        }
        return try values.map { try intValue($0, at: path) }
    }

    private static func int32Array(_ value: Any?, at path: String) throws -> [Int32] {
        try (value as? [Any] ?? []).map { try int32Value($0, at: path) }
    }

    private static func nestedInt32(_ value: Any?, at path: String) throws -> NestedInt32 {
        if let number = value as? NSNumber {
            guard let scalar = Int32(exactly: number.intValue) else {
                throw failure("reference", "\(path) contains an Int32 overflow")
            }
            return NestedInt32(shape: [], values: [scalar])
        }
        guard let children = value as? [Any], !children.isEmpty else {
            throw failure("reference", "\(path) contains an empty or invalid array")
        }
        let decoded = try children.map { try nestedInt32($0, at: path) }
        guard decoded.dropFirst().allSatisfy({ $0.shape == decoded[0].shape }) else {
            throw failure("reference", "\(path) is ragged")
        }
        return NestedInt32(
            shape: [children.count] + decoded[0].shape,
            values: decoded.flatMap(\.values))
    }

    private static func double(_ value: Any?) -> Double? {
        (value as? NSNumber)?.doubleValue
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func isDigest(_ value: String) -> Bool {
        value.count == 64 && value.unicodeScalars.allSatisfy {
            ($0.value >= 48 && $0.value <= 57)
                || ($0.value >= 97 && $0.value <= 102)
        }
    }

    private static func decodeFloat32(_ data: Data) -> [Float] {
        guard data.count.isMultiple(of: 4) else { return [] }
        return stride(from: 0, to: data.count, by: 4).map { offset in
            let bytes = Array(data[offset..<(offset + 4)])
            let bits = UInt32(bytes[0])
                | UInt32(bytes[1]) << 8
                | UInt32(bytes[2]) << 16
                | UInt32(bytes[3]) << 24
            return Float(bitPattern: bits)
        }
    }

    private static func failure(_ gate: String, _ detail: String) -> ParityError {
        print("P23_REAL_VISION_FAILURE gate=\(gate) detail=\(detail)")
        return .failure(gate: gate, detail: detail)
    }
}
