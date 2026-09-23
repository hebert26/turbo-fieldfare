import CryptoKit
import Foundation
import Darwin
import Metal
import Testing
@testable import TurboFieldfare

/// Same-pack qualification for the production Qwen text path.
///
/// This suite is deliberately opt-in. Discovery without every explicit path
/// and identity value is a disabled test, and therefore cannot download,
/// discover, or load a model as a side effect of an ordinary test run.
@Suite(.serialized) struct QwenRealTextParityTests {
    private static let vocabularySize = 248_320
    private static let eosID: Int32 = 248_044
    private static let officialModelID = "Qwen/Qwen3.6-35B-A3B"
    private static let officialSourceRevision = "995ad96eacd98c81ed38be0c5b274b04031597b0"
    private static let transformersCommit = "bd15bc95a89e728bbc1224084eb3b5829428c353"
    private static let transformersTree = "80eb369e589827bc7ed45b3a1f0ead5457097535"
    private static let modelingSourceSHA256 =
        "971b08ed3eb7452f3f5f1b0f8ab8e602fc4c4e2cc10d0a4e99ec1e1830776074"
    private static let configurationSourceSHA256 =
        "9f68bcddc54b4e512802e18ec8a242514d7f746795b28373805d3e05a981f573"
    private static let sourceIndexSHA256 =
        "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
    private static let fp32FixtureSHA256 =
        "e07372907e09f7deab72abd64f417b6f514953a098ca51d53845dc832ae6b1ec"

    private static let activationEnvironment = "TURBO_FIELDFARE_REAL_QWEN_ARTIFACT"

    @Test(.enabled(if: Self.isExplicitlyEnabled,
                   "set all explicit Qwen parity environment values to run the qualification"))
    func realQuantizedTextMatchesIndependentReference() async throws {
        let configuration = try Configuration.fromEnvironment()
        try FileManager.default.createDirectory(
            at: configuration.referenceOutput.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: configuration.parityOutput.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let manifestURL = configuration.modelDirectory.appendingPathComponent("manifest.json")
        let manifestData = try Data(contentsOf: manifestURL)
        let loadedManifest = try ManifestReader.loadVerified(directoryURL: configuration.modelDirectory)
        try Self.validateLoadedManifest(
            loadedManifest, manifestData: manifestData,
            configuration: configuration)
        let tokenizer = try QwenTokenizer.load(from: configuration.modelDirectory)
        let codec = QwenChatCodec(tokenizer: tokenizer)
        let prompts = try Self.fixedPrompts(codec: codec)

        let temporaryDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen36-real-text-parity-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }

        let requestURL = temporaryDirectory.appendingPathComponent("request.json")
        let resolvedRequest = requestURL.standardizedFileURL.resolvingSymlinksInPath().standardizedFileURL
        let resolvedOutputs = [configuration.referenceOutput, configuration.parityOutput]
            .map { $0.standardizedFileURL.resolvingSymlinksInPath().standardizedFileURL }
        guard resolvedOutputs.allSatisfy({ $0 != resolvedRequest }) else {
            throw ParityError.invalidConfiguration("output equals the generated request input")
        }
        try Self.writeRequest(Self.ReferenceRequest(cases: prompts), to: requestURL)
        // The child is fully reaped before this line returns. No MetalContext,
        // Qwen model, or runner is created before that synchronization point.
        let processResult = try await Self.runReference(
            configuration: configuration,
            requestURL: requestURL,
            outputURL: configuration.referenceOutput)
        guard processResult.status == 0 else {
            guard !Self.pathExistsIncludingDanglingSymlink(configuration.referenceOutput) else {
                throw ParityError.referenceUnexpectedOutput
            }
            let stderr = String(data: processResult.stderr, encoding: .utf8) ?? ""
            guard stderr.split(separator: "\n", omittingEmptySubsequences: true).count == 1,
                  stderr.hasPrefix("qwen36_quantized_reference: ") else {
                throw ParityError.referenceUnexpectedOutput
            }
            throw ParityError.referenceFailure(
                status: processResult.status,
                stderr: stderr)
        }
        guard processResult.stdout.isEmpty else {
            throw ParityError.referenceUnexpectedOutput
        }

        let referenceData = try Data(contentsOf: configuration.referenceOutput)
        try Self.validateExactReferenceKeys(referenceData)
        let reference = try JSONDecoder().decode(ReferenceEnvelope.self, from: referenceData)
        try Self.validateReference(reference, configuration: configuration,
                                   prompts: prompts,
                                   scriptData: try Data(contentsOf: configuration.referenceScript),
                                   officialConfigData: try Data(contentsOf: configuration.officialConfig),
                                   actualManifest: loadedManifest)

        let context = try MetalContext()
        guard case .qwen(let model) = try ModelFamilyRuntime.load(
            directoryURL: configuration.modelDirectory,
            device: context.device) else {
            throw ParityError.wrongRuntimeFamily
        }

        var candidateCases: [CandidateCase] = []
        for (referenceCase, prompt) in zip(reference.cases, prompts) {
            let runner = try model.makeRunner(context: context)
            var candidateSteps: [CandidateStep] = []
            var candidateStopReason: String?
            var candidateFailure: String?
            for (index, step) in referenceCase.steps.enumerated() {
                let result: QwenTextHybridResult
                do {
                    if index == 0 {
                        result = try await runner.prefill(tokens: prompt.tokenIDs)
                    } else {
                        guard let previous = candidateSteps.last?.comparison else {
                            candidateFailure = "candidate trajectory ended before step \(index)"
                            break
                        }
                        result = try await runner.decode(token: previous.greedyTokenID)
                    }
                } catch {
                    candidateFailure = "candidate forward step \(index) failed: \(error)"
                    break
                }
                let logits = Array(result.output.logits.suffix(Self.vocabularySize))
                let comparison: Comparison?
                let logitFailure = Self.candidateLogitFailure(logits)
                if logitFailure == nil {
                    comparison = Self.compare(
                        candidate: logits,
                        reference: try Self.decodeLogits(step.logits),
                        expectedGreedyTokenID: step.greedyTokenID)
                } else {
                    comparison = nil
                }
                candidateSteps.append(CandidateStep(
                    step: index,
                    sequenceLength: result.output.state.sequenceLength,
                    expectedSequenceLength: step.sequenceLengthAfterForward,
                    sequenceMatches: result.output.state.sequenceLength == step.sequenceLengthAfterForward,
                    logits: logits,
                    comparison: comparison,
                    failure: logitFailure))
                if let logitFailure {
                    candidateFailure = "candidate step \(index): \(logitFailure)"
                    break
                }
                guard let comparison else {
                    candidateFailure = "candidate step \(index) has no comparison"
                    break
                }
                if comparison.greedyTokenID != step.greedyTokenID {
                    candidateStopReason = comparison.greedyTokenID == Self.eosID ? "eos" : "diverged"
                    break
                }
                if comparison.greedyTokenID == Self.eosID {
                    candidateStopReason = "eos"
                    break
                }
            }
            if candidateStopReason == nil, candidateFailure == nil {
                candidateStopReason = candidateSteps.count == referenceCase.steps.count ? "maxTokens" : "diverged"
            }
            let generated = candidateSteps.compactMap { $0.comparison?.greedyTokenID }
            let resolvedStopReason = candidateStopReason ?? "failure"
            candidateCases.append(CandidateCase(
                id: referenceCase.id,
                generatedTokenIDs: generated,
                stopReason: resolvedStopReason,
                referenceStopReason: referenceCase.stopReason,
                stopReasonMatchesReference: resolvedStopReason == referenceCase.stopReason,
                failure: candidateFailure,
                matchesReference: candidateFailure == nil
                    && resolvedStopReason == referenceCase.stopReason
                    && generated == referenceCase.generatedTokenIDs
                    && candidateSteps.count == referenceCase.steps.count
                    && candidateSteps.allSatisfy { $0.sequenceMatches && ($0.comparison?.passed ?? false) },
                steps: candidateSteps))
        }

        let parityPassed = candidateCases.allSatisfy(\.matchesReference)
        let evidence = try Self.makeEvidence(
            configuration: configuration,
            referenceData: referenceData,
            requestURL: requestURL,
            referenceStderr: processResult.stderr,
            prompts: prompts,
            candidateCases: candidateCases,
            verdict: parityPassed ? "pass" : "fail")
        try Self.writeAtomically(evidence, to: configuration.parityOutput)
        guard parityPassed else {
            throw ParityError.candidateMismatch("candidate rows, greedy IDs, or sequence lengths differ")
        }
    }

    private static var isExplicitlyEnabled: Bool {
        guard let value = ProcessInfo.processInfo.environment[activationEnvironment] else { return false }
        return !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private static func pathExistsIncludingDanglingSymlink(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path)
            || (try? FileManager.default.destinationOfSymbolicLink(atPath: url.path)) != nil
    }

    private struct Configuration: Sendable {
        let modelDirectory: URL
        let officialConfig: URL
        let transformersCheckout: URL
        let referencePython: URL
        let referenceScript: URL
        let referenceOutput: URL
        let parityOutput: URL
        let manifestSHA256: String
        let policySHA256: String
        let torchThreads: Int
        let candidateDigest: String

        static func fromEnvironment() throws -> Self {
            let environment = ProcessInfo.processInfo.environment
            func required(_ name: String) throws -> String {
                guard let value = environment[name], !value.isEmpty else {
                    throw ParityError.missingEnvironment(name)
                }
                return value
            }
            let manifest = try required("TURBO_FIELDFARE_QWEN_MANIFEST_SHA256")
            let policy = try required("TURBO_FIELDFARE_QWEN_POLICY_SHA256")
            let digest = try required("TURBO_FIELDFARE_QWEN_CANDIDATE_DIGEST")
            guard Self.isDigest(manifest), Self.isDigest(policy), Self.isDigest(digest) else {
                throw ParityError.invalidConfiguration("identity values must be lowercase SHA-256")
            }
            guard let threads = Int(try required("TURBO_FIELDFARE_QWEN_TORCH_THREADS")),
                  (1...12).contains(threads) else {
                throw ParityError.invalidConfiguration("torch threads must be 1...12")
            }
            let configuration = Self(
                modelDirectory: URL(fileURLWithPath: try required(QwenRealTextParityTests.activationEnvironment), isDirectory: true),
                officialConfig: URL(fileURLWithPath: try required("TURBO_FIELDFARE_QWEN_REAL_OFFICIAL_CONFIG")),
                transformersCheckout: Self.canonical(URL(fileURLWithPath: try required("TURBO_FIELDFARE_QWEN_TRANSFORMERS_CHECKOUT"), isDirectory: true)),
                referencePython: URL(fileURLWithPath: try required("TURBO_FIELDFARE_QWEN_REFERENCE_PYTHON")),
                referenceScript: URL(fileURLWithPath: try required("TURBO_FIELDFARE_QWEN_REFERENCE_SCRIPT")),
                referenceOutput: URL(fileURLWithPath: try required("TURBO_FIELDFARE_QWEN_REFERENCE_OUTPUT")),
                parityOutput: URL(fileURLWithPath: try required("TURBO_FIELDFARE_QWEN_PARITY_OUTPUT")),
                manifestSHA256: manifest,
                policySHA256: policy,
                torchThreads: threads,
                candidateDigest: digest)
            try configuration.validatePaths()
            return configuration
        }

        private func validatePaths() throws {
            let model = Self.canonical(modelDirectory)
            let transformers = Self.canonical(transformersCheckout)
            let outputs = [referenceOutput, parityOutput].map(Self.canonical)
            let repository = Self.canonical(URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
            let forbiddenRoots = ["Sources", "Tests", "Scripts"].map {
                Self.canonical(repository.appendingPathComponent($0, isDirectory: true))
            }
            guard outputs[0] != outputs[1] else {
                throw ParityError.invalidConfiguration("reference and parity outputs must be distinct")
            }
            guard FileManager.default.fileExists(atPath: transformers.path) else {
                throw ParityError.invalidConfiguration("Transformers checkout does not exist")
            }
            let inputs = [modelDirectory, officialConfig, transformersCheckout, referencePython, referenceScript]
            for output in outputs {
                guard !Self.isWithin(output, model) else {
                    throw ParityError.invalidConfiguration("output is inside the model artifact")
                }
                guard !Self.isWithin(output, transformers) else {
                    throw ParityError.invalidConfiguration("output is inside the Transformers checkout")
                }
                guard !forbiddenRoots.contains(where: { Self.isWithin(output, $0) }) else {
                    throw ParityError.invalidConfiguration("output is inside repository source or test inputs")
                }
                guard !inputs.map(Self.canonical).contains(output) else {
                    throw ParityError.invalidConfiguration("output equals an input")
                }
                guard !QwenRealTextParityTests.pathExistsIncludingDanglingSymlink(output) else {
                    throw ParityError.invalidConfiguration("output must be a fresh nonexistent path")
                }
            }
        }

        private static func canonical(_ url: URL) -> URL {
            url.standardizedFileURL.resolvingSymlinksInPath().standardizedFileURL
        }

        private static func isWithin(_ child: URL, _ parent: URL) -> Bool {
            child.path == parent.path || child.path.hasPrefix(parent.path + "/")
        }


        private static func isDigest(_ value: String) -> Bool {
            value.count == 64 && value.unicodeScalars.allSatisfy { scalar in
                (scalar.value >= 48 && scalar.value <= 57)
                    || (scalar.value >= 97 && scalar.value <= 102)
            }
        }
    }

    private struct FixedPrompt: Codable, Sendable {
        let id: String
        let tokenIDs: [Int32]
        let maxNewTokens: Int
        let text: String
    }

    private static func fixedPrompts(codec: QwenChatCodec) throws -> [FixedPrompt] {
        let source = [("ordinary-1", "Return OK."), ("ordinary-2", "Name one color.")]
        return try source.map { id, text in
            let tokens = try codec.encodePrompt(
                messages: [ModelChatMessage(role: .user, content: text)],
                tools: [],
                options: ModelChatRenderOptions(
                    addGenerationPrompt: true,
                    enableThinking: false,
                    preserveThinking: false,
                    addVisionID: false))
            guard (1...32).contains(tokens.count),
                  tokens.allSatisfy({ $0 >= 0 && $0 < Int32(Self.vocabularySize) }) else {
                throw ParityError.invalidPrompt(id)
            }
            return FixedPrompt(id: id, tokenIDs: tokens, maxNewTokens: 2, text: text)
        }
    }

    private struct ReferenceRequest: Encodable {
        let schemaVersion = "qwen36-quantized-reference-request-v1"
        let cases: [RequestCase]

        init(cases: [FixedPrompt]) {
            self.cases = cases.map { RequestCase(id: $0.id, promptTokenIDs: $0.tokenIDs, maxNewTokens: $0.maxNewTokens) }
        }
    }

    private struct RequestCase: Encodable {
        let id: String
        let promptTokenIDs: [Int32]
        let maxNewTokens: Int
    }

    private static func writeRequest(_ request: ReferenceRequest, to url: URL) throws {
        let data = try JSONEncoder().encode(request)
        try writeAtomically(data, to: url)
    }

    private final class BoundedCapture: @unchecked Sendable {
        private let lock = NSLock()
        private let maximumBytes: Int
        private var data = Data()
        private var truncated = false

        init(maximumBytes: Int = 64 * 1024) { self.maximumBytes = maximumBytes }

        func append(_ bytes: Data) {
            lock.withLock {
                let remaining = maximumBytes - data.count
                if remaining > 0 { data.append(bytes.prefix(remaining)) }
                if bytes.count > max(remaining, 0) { truncated = true }
            }
        }

        func snapshot() -> (data: Data, truncated: Bool) {
            lock.withLock { (data, truncated) }
        }
    }

    private final class OwnedReferenceProcess: @unchecked Sendable {
        private let lock = NSLock()
        private var process: Process?
        private var cancelled = false
        private var cancellationSignal: DispatchSemaphore?

        func install(_ process: Process) {
            lock.withLock {
                self.process = process
                if cancelled, process.isRunning { process.terminate() }
            }
        }
        func clear() { lock.withLock { process = nil; cancellationSignal = nil } }
        func installCancellationSignal(_ signal: DispatchSemaphore) {
            lock.withLock {
                cancellationSignal = signal
                if cancelled { signal.signal() }
            }
        }
        func terminate() {
            lock.withLock {
                cancelled = true
                cancellationSignal?.signal()
                guard let process, process.isRunning else { return }
                process.terminate()
            }
        }
        var wasCancelled: Bool { lock.withLock { cancelled } }
    }

    private struct ProcessResult: Sendable {
        let status: Int32
        let stdout: Data
        let stderr: Data
    }

    private static func runReference(
        configuration: Configuration,
        requestURL: URL,
        outputURL: URL
    ) async throws -> ProcessResult {
        let owner = OwnedReferenceProcess()
        return try await withTaskCancellationHandler(operation: {
            let result = try runReferenceBlocking(
                configuration: configuration,
                requestURL: requestURL,
                outputURL: outputURL,
                owner: owner)
            try Task.checkCancellation()
            return result
        }, onCancel: {
            owner.terminate()
        })
    }

    private static func runReferenceBlocking(
        configuration: Configuration,
        requestURL: URL,
        outputURL: URL,
        owner: OwnedReferenceProcess
    ) throws -> ProcessResult {
        let process = Process()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        let stdoutCapture = BoundedCapture()
        let stderrCapture = BoundedCapture()
        let drainGroup = DispatchGroup()
        func drain(_ handle: FileHandle, into capture: BoundedCapture) {
            drainGroup.enter()
            DispatchQueue.global(qos: .utility).async {
                defer { drainGroup.leave() }
                while true {
                    guard let bytes = try? handle.read(upToCount: 16 * 1024), !bytes.isEmpty else { break }
                    capture.append(bytes)
                }
            }
        }
        process.executableURL = configuration.referencePython
        process.arguments = [
            configuration.referenceScript.path,
            "--model-directory", configuration.modelDirectory.path,
            "--official-config", configuration.officialConfig.path,
            "--transformers-checkout", configuration.transformersCheckout.path,
            "--request", requestURL.path,
            "--expected-manifest-sha256", configuration.manifestSHA256,
            "--expected-policy-sha256", configuration.policySHA256,
            "--torch-threads", String(configuration.torchThreads),
            "--output", outputURL.path,
        ]
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONHASHSEED"] = "0"
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["USE_HUB_KERNELS"] = "0"
        process.environment = environment
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        drain(stdoutPipe.fileHandleForReading, into: stdoutCapture)
        drain(stderrPipe.fileHandleForReading, into: stderrCapture)
        var outputsFinished = false
        defer {
            owner.clear()
            if !outputsFinished {
                stdoutPipe.fileHandleForWriting.closeFile()
                stderrPipe.fileHandleForWriting.closeFile()
                stdoutPipe.fileHandleForReading.closeFile()
                stderrPipe.fileHandleForReading.closeFile()
                _ = drainGroup.wait(timeout: .now() + 5)
            }
        }
        owner.install(process)
        guard !owner.wasCancelled else { throw CancellationError() }
        let cancellationSignal = DispatchSemaphore(value: 0)
        owner.installCancellationSignal(cancellationSignal)
        process.terminationHandler = { _ in cancellationSignal.signal() }
        func finishOutputDrains() throws {
            stdoutPipe.fileHandleForWriting.closeFile()
            stderrPipe.fileHandleForWriting.closeFile()
            guard drainGroup.wait(timeout: .now() + 5) == .success else {
                stdoutPipe.fileHandleForReading.closeFile()
                stderrPipe.fileHandleForReading.closeFile()
                guard drainGroup.wait(timeout: .now() + 5) == .success else {
                    throw ParityError.referenceCleanupFailed
                }
                throw ParityError.referenceCleanupFailed
            }
            stdoutPipe.fileHandleForReading.closeFile()
            stderrPipe.fileHandleForReading.closeFile()
            outputsFinished = true
        }
        do {
            try process.run()
        } catch {
            throw ParityError.referenceLaunch(String(describing: error))
        }
        stdoutPipe.fileHandleForWriting.closeFile()
        stderrPipe.fileHandleForWriting.closeFile()
        if owner.wasCancelled { owner.terminate() }
        let deadline = Date().addingTimeInterval(15 * 60)
        var cancellationRequested = false
        while process.isRunning && Date() < deadline {
            if cancellationSignal.wait(timeout: .now() + 0.25) == .success {
                cancellationRequested = owner.wasCancelled
                break
            }
        }
        if cancellationRequested || process.isRunning {
            owner.terminate()
            let graceDeadline = Date().addingTimeInterval(5)
            while process.isRunning && Date() < graceDeadline {
                _ = cancellationSignal.wait(timeout: .now() + 0.25)
            }
            if process.isRunning { _ = kill(process.processIdentifier, SIGKILL) }
            process.waitUntilExit()
            try finishOutputDrains()
            if cancellationRequested || owner.wasCancelled { throw CancellationError() }
            throw ParityError.referenceTimeout
        }
        process.waitUntilExit()
        try finishOutputDrains()
        guard !owner.wasCancelled else { throw CancellationError() }
        let stdout = stdoutCapture.snapshot()
        let stderr = stderrCapture.snapshot()
        guard !stdout.truncated, !stderr.truncated else {
            throw ParityError.referenceOutputTooLarge
        }
        return ProcessResult(status: process.terminationStatus, stdout: stdout.data, stderr: stderr.data)
    }

    private struct ReferenceEnvelope: Decodable {
        let schemaVersion: String
        let status: String
        let identity: ReferenceIdentity
        let referenceEnvironment: ReferenceEnvironment
        let reader: ReferenceReader
        let tolerances: ReferenceTolerances
        let cases: [ReferenceCase]
    }

    private struct ReferenceIdentity: Decodable {
        let manifestSHA256: String
        let quantizationPolicySHA256: String
        let modelID: String
        let sourceRevision: String
        let sourceIndexSHA256: String
        let configSHA256: String
        let files: [ReferenceFile]
    }

    private struct ReferenceFile: Decodable {
        let path: String
        let size: UInt64
        let sha256: String
    }

    private struct ReferenceEnvironment: Decodable {
        let pythonVersion: String
        let torchVersion: String
        let transformersVersion: String
        let numpyVersion: String
        let transformersCommit: String
        let transformersTree: String
        let modelingSourceSHA256: String
        let configurationSourceSHA256: String
        let platform: String
        let processor: String
        let device: String
        let dtype: String
        let attentionImplementation: String
        let expertsImplementation: String
        let offline: ReferenceOffline
        let deterministicAlgorithms: Bool
        let torchThreads: Int
        let torchInteropThreads: Int
    }

    private struct ReferenceOffline: Decodable {
        let hashSeed: String
        let hubOffline: String
        let transformersOffline: String
        let useHubKernels: String

        enum CodingKeys: String, CodingKey {
            case hashSeed = "PYTHONHASHSEED"
            case hubOffline = "HF_HUB_OFFLINE"
            case transformersOffline = "TRANSFORMERS_OFFLINE"
            case useHubKernels = "USE_HUB_KERNELS"
        }
    }

    private struct ReferenceReader: Decodable {
        let schema: String
        let scriptSHA256: String
        let maximumTransientDecodeBytes: UInt64
        let maximumLiveDecodedLayerBytes: UInt64
        let runtimeModulesImported: Bool
        let bf16CheckpointRead: Bool
    }

    private struct ReferenceTolerances: Decodable {
        let exactInteger: Pair
        let fp32: FP32
        let stateFP32: StateFP32

        struct Pair: Decodable { let absolute: Double; let relative: Double }
        struct FP32: Decodable {
            let absolute: Double
            let relative: Double
            let comparisonRule: String
            let source: String
            let fixtureSHA256: String
        }
        struct StateFP32: Decodable {
            let absolute: Double
            let relative: Double
            let source: String
        }
    }

    private struct ReferenceCase: Decodable {
        let id: String
        let promptTokenIDs: [Int32]
        let promptTokenIDSHA256: String
        let maxNewTokens: Int
        let generatedTokenIDs: [Int32]
        let stopReason: String
        let steps: [ReferenceStep]
    }

    private struct ReferenceStep: Decodable {
        let step: Int
        let inputTokenIDs: [Int32]
        let sequenceLengthAfterForward: Int
        let logits: ReferenceLogits
        let greedyTokenID: Int32
        let topK: [ReferenceTopK]
    }

    private struct ReferenceLogits: Decodable {
        let dtype: String
        let encoding: String
        let count: Int
        let sha256: String
        let bytes: String
    }

    private struct ReferenceTopK: Decodable, Equatable {
        let tokenID: Int32
        let logit: Float
    }

    private static func validateExactReferenceKeys(_ data: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ParityError.invalidReference("result is not an object")
        }
        try exact(root, keys: ["schemaVersion", "status", "identity", "referenceEnvironment", "reader", "tolerances", "cases"], path: "result")
        let identity = try object(root["identity"], path: "identity")
        try exact(identity, keys: ["manifestSHA256", "quantizationPolicySHA256", "modelID", "sourceRevision", "sourceIndexSHA256", "configSHA256", "files"], path: "identity")
        for (index, value) in try array(identity["files"], path: "identity.files").enumerated() {
            let file = try object(value, path: "identity.files[\(index)]")
            try exact(file, keys: ["path", "size", "sha256"], path: "identity.files[\(index)]")
        }
        let environment = try object(root["referenceEnvironment"], path: "referenceEnvironment")
        try exact(environment, keys: ["pythonVersion", "torchVersion", "transformersVersion", "numpyVersion", "transformersCommit", "transformersTree", "modelingSourceSHA256", "configurationSourceSHA256", "platform", "processor", "device", "dtype", "attentionImplementation", "expertsImplementation", "offline", "deterministicAlgorithms", "torchThreads", "torchInteropThreads"], path: "referenceEnvironment")
        let offline = try object(environment["offline"], path: "referenceEnvironment.offline")
        try exact(offline, keys: ["PYTHONHASHSEED", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "USE_HUB_KERNELS"], path: "referenceEnvironment.offline")
        let reader = try object(root["reader"], path: "reader")
        try exact(reader, keys: ["schema", "scriptSHA256", "maximumTransientDecodeBytes", "maximumLiveDecodedLayerBytes", "runtimeModulesImported", "bf16CheckpointRead"], path: "reader")
        let tolerances = try object(root["tolerances"], path: "tolerances")
        try exact(tolerances, keys: ["exactInteger", "fp32", "stateFP32"], path: "tolerances")
        for key in ["exactInteger", "stateFP32"] {
            let pair = try object(tolerances[key], path: "tolerances.\(key)")
            if key == "exactInteger" {
                try exact(pair, keys: ["absolute", "relative"], path: "tolerances.\(key)")
            } else {
                try exact(pair, keys: ["absolute", "relative", "source"], path: "tolerances.\(key)")
            }
        }
        let fp32 = try object(tolerances["fp32"], path: "tolerances.fp32")
        try exact(fp32, keys: ["absolute", "relative", "comparisonRule", "source", "fixtureSHA256"], path: "tolerances.fp32")
        for (index, value) in try array(root["cases"], path: "cases").enumerated() {
            let item = try object(value, path: "cases[\(index)]")
            try exact(item, keys: ["id", "promptTokenIDs", "promptTokenIDSHA256", "maxNewTokens", "generatedTokenIDs", "stopReason", "steps"], path: "cases[\(index)]")
            for (stepIndex, stepValue) in try array(item["steps"], path: "cases[\(index)].steps").enumerated() {
                let step = try object(stepValue, path: "cases[\(index)].steps[\(stepIndex)]")
                try exact(step, keys: ["step", "inputTokenIDs", "sequenceLengthAfterForward", "logits", "greedyTokenID", "topK"], path: "cases[\(index)].steps[\(stepIndex)]")
                let logits = try object(step["logits"], path: "cases[\(index)].steps[\(stepIndex)].logits")
                try exact(logits, keys: ["dtype", "encoding", "count", "sha256", "bytes"], path: "cases[\(index)].steps[\(stepIndex)].logits")
                for (topIndex, topValue) in try array(step["topK"], path: "cases[\(index)].steps[\(stepIndex)].topK").enumerated() {
                    let top = try object(topValue, path: "topK[\(topIndex)]")
                    try exact(top, keys: ["tokenID", "logit"], path: "topK[\(topIndex)]")
                }
            }
        }
    }

    private static func object(_ value: Any?, path: String) throws -> [String: Any] {
        guard let value, let object = value as? [String: Any] else {
            throw ParityError.invalidReference("\(path) is not an object")
        }
        return object
    }

    private static func array(_ value: Any?, path: String) throws -> [Any] {
        guard let value, let array = value as? [Any] else {
            throw ParityError.invalidReference("\(path) is not an array")
        }
        return array
    }

    private static func exact(_ object: [String: Any], keys: Set<String>, path: String) throws {
        guard Set(object.keys) == keys else {
            throw ParityError.invalidReference("unexpected or missing keys at \(path)")
        }
    }

    private static func validateLoadedManifest(
        _ manifest: LoadedModelManifest,
        manifestData: Data,
        configuration: Configuration
    ) throws {
        guard sha256(manifestData) == configuration.manifestSHA256,
              manifest.descriptor.textManifestSHA256 == configuration.manifestSHA256,
              manifest.descriptor.family == .qwen3_6,
              manifest.descriptor.modelID == officialModelID,
              manifest.descriptor.sourceRevision == officialSourceRevision,
              manifest.descriptor.sourceIndexSHA256 == sourceIndexSHA256,
              manifest.descriptor.quantizationPolicySHA256 == configuration.policySHA256,
              manifest.files.keys.count == Set(manifest.files.keys).count,
              manifest.tensorRegions.count == Set(manifest.tensorRegions.map(\.name)).count,
              manifest.files.keys.allSatisfy({ !$0.isEmpty && !$0.hasPrefix("/") }),
              manifest.tensorRegions.allSatisfy({ !$0.name.isEmpty && !$0.file.hasPrefix("/") }) else {
            throw ParityError.invalidReference("actual manifest identity or inventory")
        }
        let filePaths = Set(manifest.files.keys)
        guard manifest.tensorRegions.allSatisfy({ filePaths.contains($0.file) }) else {
            throw ParityError.invalidReference("manifest tensor region references an unlisted file")
        }
        guard case .qwen3_6(let architecture) = manifest.architecture,
              architecture.vocabularySize == vocabularySize,
              architecture.eosTokenID == Int(eosID),
              architecture.expertsPerToken == 8,
              architecture.numLayers == manifest.numLayers,
              architecture.numberOfExperts == 256,
              manifest.expertsPerLayer == 256,
              manifest.expertStride > 0 else {
            throw ParityError.invalidReference("actual Qwen architecture")
        }
    }

    private static func validateReference(
        _ reference: ReferenceEnvelope,
        configuration: Configuration,
        prompts: [FixedPrompt],
        scriptData: Data,
        officialConfigData: Data,
        actualManifest: LoadedModelManifest
    ) throws {
        guard reference.schemaVersion == "qwen36-quantized-reference-v1",
              reference.status == "complete" else {
            throw ParityError.invalidReference("schema or status")
        }
        let identity = reference.identity
        guard identity.manifestSHA256 == configuration.manifestSHA256,
              identity.quantizationPolicySHA256 == configuration.policySHA256,
              identity.modelID == officialModelID,
              identity.sourceRevision == officialSourceRevision,
              identity.sourceIndexSHA256 == actualManifest.descriptor.sourceIndexSHA256,
              identity.files.map(\.path) == identity.files.map(\.path).sorted(),
              Self.isDigest(identity.sourceIndexSHA256),
              identity.configSHA256 == Self.sha256(officialConfigData) else {
            throw ParityError.invalidReference("identity")
        }
        guard identity.files.allSatisfy({
            !$0.path.isEmpty && !$0.path.hasPrefix("/") && !$0.path.split(separator: "/").contains("..")
                && $0.size > 0 && Self.isDigest($0.sha256)
        }) else {
            throw ParityError.invalidReference("identity files")
        }
        guard identity.files.count == actualManifest.files.count,
              Set(identity.files.map(\.path)) == Set(actualManifest.files.keys),
              identity.files.allSatisfy({ file in
                  guard let actual = actualManifest.files[file.path] else { return false }
                  return actual.size == file.size && actual.sha256.lowercased() == file.sha256
              }) else {
            throw ParityError.invalidReference("reference file inventory does not equal manifest")
        }
        let environment = reference.referenceEnvironment
        guard environment.pythonVersion == "3.12.3",
              environment.torchVersion == "2.10.0",
              environment.transformersVersion == "5.18.0.dev0",
              environment.numpyVersion == "2.4.3",
              environment.transformersCommit == transformersCommit,
              environment.transformersTree == transformersTree,
              environment.modelingSourceSHA256 == modelingSourceSHA256,
              environment.configurationSourceSHA256 == configurationSourceSHA256,
              environment.platform == "Darwin", environment.device == "cpu",
              environment.dtype == "float32",
              environment.attentionImplementation == "eager",
              environment.expertsImplementation == "eager",
              environment.offline.hashSeed == "0",
              environment.offline.hubOffline == "1",
              environment.offline.transformersOffline == "1",
              environment.offline.useHubKernels == "0",
              environment.deterministicAlgorithms,
              environment.torchThreads == configuration.torchThreads,
              environment.torchInteropThreads == 1 else {
            throw ParityError.invalidReference("reference environment")
        }
        let reader = reference.reader
        guard reader.schema == "gturbo-v2-qwen-text-independent-v1",
              reader.scriptSHA256 == Self.sha256(scriptData),
              reader.maximumTransientDecodeBytes > 0,
              reader.maximumTransientDecodeBytes == 64 * 1024 * 1024,
              reader.maximumLiveDecodedLayerBytes > 0,
              reader.maximumLiveDecodedLayerBytes < 4 * 1024 * 1024 * 1024,
              !reader.runtimeModulesImported, !reader.bf16CheckpointRead else {
            throw ParityError.invalidReference("reader provenance")
        }
        let tolerances = reference.tolerances
        guard tolerances.exactInteger.absolute == 0,
              tolerances.exactInteger.relative == 0,
              tolerances.fp32.absolute == 0.00001,
              tolerances.fp32.relative == 0.00001,
              tolerances.fp32.comparisonRule ==
                "maxAbs <= absolute + relative * max(maxAbsReference,maxAbsCandidate)",
              tolerances.fp32.source == "fixed before Swift runtime implementation; packed-byte CPU oracle",
              tolerances.fp32.fixtureSHA256 == fp32FixtureSHA256,
              tolerances.stateFP32.absolute == 0.00002,
              tolerances.stateFP32.relative == 0.00002,
              tolerances.stateFP32.source == "fixed before Swift runtime implementation; chunk/recurrent reassociation" else {
            throw ParityError.invalidReference("frozen tolerances")
        }
        guard reference.cases.count == prompts.count else {
            throw ParityError.invalidReference("case count")
        }
        for (item, prompt) in zip(reference.cases, prompts) {
            guard item.id == prompt.id,
                  item.promptTokenIDs == prompt.tokenIDs,
                  item.promptTokenIDSHA256 == Self.sha256(Self.int32Data(prompt.tokenIDs)),
                  item.maxNewTokens == prompt.maxNewTokens,
                  !item.steps.isEmpty,
                  item.steps.count <= prompt.maxNewTokens else {
                throw ParityError.invalidReference("case \(prompt.id)")
            }
            for (index, step) in item.steps.enumerated() {
                let expectedInput = index == 0 ? prompt.tokenIDs : [item.steps[index - 1].greedyTokenID]
                let decodedLogits = try Self.decodeLogits(step.logits)
                guard step.step == index,
                      step.inputTokenIDs == expectedInput,
                      step.sequenceLengthAfterForward == prompt.tokenIDs.count + index,
                      step.topK.count == 5,
                      step.topK.allSatisfy({ $0.logit.isFinite && (0..<Int32(vocabularySize)).contains($0.tokenID) }),
                      step.topK == step.topK.sorted(by: Self.topKPrecedes),
                      step.topK.first?.tokenID == Optional(step.greedyTokenID),
                      decodedLogits.count == vocabularySize,
                      Self.firstArgmax(decodedLogits) == step.greedyTokenID,
                      step.topK == Self.expectedTopK(decodedLogits) else {
                    throw ParityError.invalidReference("steps \(prompt.id)/\(index)")
                }
                try validateLogitRecord(step.logits)
            }
            let generated = item.steps.map(\.greedyTokenID)
            let eosIndices = generated.enumerated().compactMap { $0.element == eosID ? $0.offset : nil }
            guard item.generatedTokenIDs == generated,
                  item.stopReason == "eos" || item.stopReason == "maxTokens",
                  eosIndices.isEmpty || eosIndices == [generated.count - 1] else {
                throw ParityError.invalidReference("termination \(prompt.id)")
            }
            if item.stopReason == "eos" {
                guard item.generatedTokenIDs.last == eosID else {
                    throw ParityError.invalidReference("eos termination \(prompt.id)")
                }
            } else {
                guard eosIndices.isEmpty, item.steps.count == item.maxNewTokens else {
                    throw ParityError.invalidReference("max-token termination \(prompt.id)")
                }
            }
        }
    }

    private static func validateLogitRecord(_ logits: ReferenceLogits) throws {
        guard logits.dtype == "float32-le", logits.encoding == "base64",
              logits.count == vocabularySize,
              let data = Data(base64Encoded: logits.bytes), data.count == vocabularySize * 4,
              Self.sha256(data) == logits.sha256 else {
            throw ParityError.invalidReference("logit bytes")
        }
        guard decodeFloat32(data).allSatisfy(\.isFinite) else {
            throw ParityError.invalidReference("non-finite logits")
        }
    }

    private static func topKPrecedes(_ left: ReferenceTopK, _ right: ReferenceTopK) -> Bool {
        if left.logit == right.logit { return left.tokenID < right.tokenID }
        return left.logit > right.logit
    }

    private static func expectedTopK(_ values: [Float]) -> [ReferenceTopK] {
        values.enumerated()
            .sorted { left, right in
                if left.element == right.element { return left.offset < right.offset }
                return left.element > right.element
            }
            .prefix(5)
            .map { ReferenceTopK(tokenID: Int32($0.offset), logit: $0.element) }
    }

    private struct Comparison: Sendable {
        let maximumAbsoluteError: Float
        let scale: Float
        let maximumRelativeError: Float
        let allowedMaximumAbsoluteError: Float
        let greedyTokenID: Int32
        let passed: Bool
    }

    private static func candidateLogitFailure(_ values: [Float]) -> String? {
        guard values.count == vocabularySize else {
            return "logit count \(values.count), expected \(vocabularySize)"
        }
        guard values.allSatisfy(\.isFinite) else {
            return "logits contain a non-finite value"
        }
        return nil
    }

    private static func compare(
        candidate: [Float], reference: [Float], expectedGreedyTokenID: Int32
    ) -> Comparison {
        var maximum: Float = 0
        var scale: Float = 0
        for (left, right) in zip(candidate, reference) {
            maximum = max(maximum, abs(left - right))
            scale = max(scale, max(abs(left), abs(right)))
        }
        let allowed = 1e-5 + 1e-5 * scale
        let greedy = Self.firstArgmax(candidate)
        return Comparison(
            maximumAbsoluteError: maximum,
            scale: scale,
            maximumRelativeError: scale == 0 ? 0 : maximum / scale,
            allowedMaximumAbsoluteError: allowed,
            greedyTokenID: greedy,
            passed: greedy == expectedGreedyTokenID && maximum <= allowed)
    }

    private static func firstArgmax(_ values: [Float]) -> Int32 {
        var index = 0
        for candidate in values.indices.dropFirst() where values[candidate] > values[index] {
            index = candidate
        }
        return Int32(index)
    }

    private struct CandidateStep {
        let step: Int
        let sequenceLength: Int
        let expectedSequenceLength: Int
        let sequenceMatches: Bool
        let logits: [Float]
        let comparison: Comparison?
        let failure: String?
    }

    private struct CandidateCase {
        let id: String
        let generatedTokenIDs: [Int32]
        let stopReason: String
        let referenceStopReason: String
        let stopReasonMatchesReference: Bool
        let failure: String?
        let matchesReference: Bool
        let steps: [CandidateStep]
    }

    private static func makeEvidence(
        configuration: Configuration,
        referenceData: Data,
        requestURL: URL,
        referenceStderr: Data,
        prompts: [FixedPrompt],
        candidateCases: [CandidateCase],
        verdict: String
    ) throws -> Data {
        guard let referenceObject = try JSONSerialization.jsonObject(with: referenceData) as? [String: Any] else {
            throw ParityError.invalidReference("reference JSON object")
        }
        let swiftCases: [[String: Any]] = candidateCases.map { item in
            let steps: [[String: Any]] = item.steps.map { step in
                var body: [String: Any] = [
                    "step": step.step,
                    "sequenceLengthAfterForward": step.sequenceLength,
                    "expectedSequenceLength": step.expectedSequenceLength,
                    "logits": Self.logitEvidence(step.logits),
                    "pass": step.sequenceMatches && (step.comparison?.passed ?? false),
                ]
                if let comparison = step.comparison {
                    body["maximumAbsoluteError"] = comparison.maximumAbsoluteError
                    body["scale"] = comparison.scale
                    body["maximumRelativeError"] = comparison.maximumRelativeError
                    body["allowedMaximumAbsoluteError"] = comparison.allowedMaximumAbsoluteError
                    body["greedyTokenID"] = Int(comparison.greedyTokenID)
                }
                if let failure = step.failure {
                    body["failure"] = failure
                }
                return body
            }
            var body: [String: Any] = [
                "id": item.id,
                "generatedTokenIDs": item.generatedTokenIDs.map(Int.init),
                "stopReason": item.stopReason,
                "referenceStopReason": item.referenceStopReason,
                "stopReasonMatchesReference": item.stopReasonMatchesReference,
                "matchesReference": item.matchesReference,
                "steps": steps,
            ]
            if let failure = item.failure {
                body["failureCategory"] = "candidate"
                body["failure"] = failure
            }
            return body
        }
        let qualificationCommand: [String] = [
            "Scripts/test.sh", "--filter", "QwenRealTextParityTests"
        ]
        let referenceCommand: [String] = [
            configuration.referencePython.path,
            configuration.referenceScript.path,
            "--model-directory", configuration.modelDirectory.path,
            "--official-config", configuration.officialConfig.path,
            "--transformers-checkout", configuration.transformersCheckout.path,
            "--request", requestURL.path,
            "--expected-manifest-sha256", configuration.manifestSHA256,
            "--expected-policy-sha256", configuration.policySHA256,
            "--torch-threads", String(configuration.torchThreads),
            "--output", configuration.referenceOutput.path,
        ]
        let promptRecords: [[String: Any]] = prompts.map { prompt in
            [
                "id": prompt.id,
                "text": prompt.text,
                "promptTokenIDs": prompt.tokenIDs.map(Int.init),
                "maxNewTokens": prompt.maxNewTokens,
            ]
        }
        let stepCount = candidateCases.reduce(0) { $0 + $1.steps.count }
        let candidateBody: [String: Any] = [
            "digest": configuration.candidateDigest,
            "implementation": "TurboFieldfare.ModelFamilyRuntime.qwen/QwenTextRunner",
            "modelDirectory": configuration.modelDirectory.path,
            "transformersCheckout": configuration.transformersCheckout.path,
            "qualificationCommand": qualificationCommand,
            "qualificationOutcome": verdict,
            "qualificationExitStatus": NSNull(),
            "qualificationExitStatusSource": "outer-harness",
            "referenceCommand": referenceCommand,
            "referenceExitStatus": 0,
            "referenceStderr": String(data: referenceStderr, encoding: .utf8) ?? "<non-UTF8>",
            "referenceOutput": configuration.referenceOutput.path,
            "parityOutput": configuration.parityOutput.path,
            "manifestSHA256": configuration.manifestSHA256,
            "quantizationPolicySHA256": configuration.policySHA256,
            "prompts": promptRecords,
            "cases": swiftCases,
            "caseCount": candidateCases.count,
            "stepCount": stepCount,
            "verdict": verdict,
        ]
        let body: [String: Any] = [
            "schemaVersion": "qwen36-real-text-parity-v1",
            "reference": referenceObject,
            "candidate": candidateBody,
        ]
        return try JSONSerialization.data(withJSONObject: body, options: [.prettyPrinted, .sortedKeys])
    }

    private static func logitEvidence(_ values: [Float]) -> [String: Any] {
        let data = float32Data(values)
        return [
            "dtype": "float32-le",
            "encoding": "base64",
            "count": values.count,
            "sha256": sha256(data),
            "bytes": data.base64EncodedString(),
        ]
    }

    private static func decodeLogits(_ logits: ReferenceLogits) throws -> [Float] {
        guard let data = Data(base64Encoded: logits.bytes) else {
            throw ParityError.invalidReference("invalid logits base64")
        }
        return decodeFloat32(data)
    }

    private static func decodeFloat32(_ data: Data) -> [Float] {
        let bytes = Array(data)
        guard bytes.count.isMultiple(of: 4) else { return [] }
        return stride(from: 0, to: bytes.count, by: 4).map { offset in
            let bits = UInt32(bytes[offset])
                | (UInt32(bytes[offset + 1]) << 8)
                | (UInt32(bytes[offset + 2]) << 16)
                | (UInt32(bytes[offset + 3]) << 24)
            return Float(bitPattern: bits)
        }
    }

    private static func float32Data(_ values: [Float]) -> Data {
        var data = Data(capacity: values.count * 4)
        for value in values {
            let bits = value.bitPattern
            data.append(UInt8(truncatingIfNeeded: bits))
            data.append(UInt8(truncatingIfNeeded: bits >> 8))
            data.append(UInt8(truncatingIfNeeded: bits >> 16))
            data.append(UInt8(truncatingIfNeeded: bits >> 24))
        }
        return data
    }

    private static func int32Data(_ values: [Int32]) -> Data {
        var data = Data(capacity: values.count * 4)
        for value in values {
            let bits = UInt32(bitPattern: value)
            data.append(UInt8(truncatingIfNeeded: bits))
            data.append(UInt8(truncatingIfNeeded: bits >> 8))
            data.append(UInt8(truncatingIfNeeded: bits >> 16))
            data.append(UInt8(truncatingIfNeeded: bits >> 24))
        }
        return data
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func isDigest(_ value: String) -> Bool {
        value.count == 64 && value.unicodeScalars.allSatisfy { scalar in
            (scalar.value >= 48 && scalar.value <= 57)
                || (scalar.value >= 97 && scalar.value <= 102)
        }
    }

    private static func writeAtomically(_ data: Data, to url: URL) throws {
        let parent = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        guard !FileManager.default.fileExists(atPath: url.path) else {
            throw ParityError.outputAlreadyExists(url.path)
        }
        let temporary = parent.appendingPathComponent(
            ".\(url.lastPathComponent).\(UUID().uuidString).tmp")
        defer { try? FileManager.default.removeItem(at: temporary) }
        try data.write(to: temporary, options: [.atomic])
        try FileManager.default.linkItem(at: temporary, to: url)
    }

    private enum ParityError: Error, CustomStringConvertible {
        case missingEnvironment(String)
        case invalidConfiguration(String)
        case invalidPrompt(String)
        case referenceLaunch(String)
        case referenceTimeout
        case referenceFailure(status: Int32, stderr: String)
        case referenceUnexpectedOutput
        case referenceOutputTooLarge
        case referenceCleanupFailed
        case outputAlreadyExists(String)
        case invalidReference(String)
        case wrongRuntimeFamily
        case candidateMismatch(String)

        var description: String {
            switch self {
            case .missingEnvironment(let name): return "missing environment \(name)"
            case .invalidConfiguration(let detail): return "invalid configuration: \(detail)"
            case .invalidPrompt(let id): return "invalid prompt \(id)"
            case .referenceLaunch(let detail): return "reference launch failed: \(detail)"
            case .referenceTimeout: return "reference process exceeded the bounded 15 minute run"
            case .referenceFailure(let status, let stderr): return "reference exited \(status): \(stderr)"
            case .referenceUnexpectedOutput: return "reference emitted unexpected stdout or stderr"
            case .referenceOutputTooLarge: return "reference stdout or stderr exceeded 64 KiB"
            case .referenceCleanupFailed: return "reference child or output drains could not be fully reaped"
            case .outputAlreadyExists(let path): return "output already exists: \(path)"
            case .invalidReference(let detail): return "invalid reference: \(detail)"
            case .wrongRuntimeFamily: return "model admitted as a non-Qwen runtime"
            case .candidateMismatch(let detail): return "candidate parity mismatch: \(detail)"
            }
        }
    }
}
