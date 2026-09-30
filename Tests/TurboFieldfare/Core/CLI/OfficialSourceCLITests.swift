import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareCLICore
@testable import TurboFieldfareOfficialQwenSource

@Suite(.serialized) struct OfficialSourceCLITests {
    @Test func integrityFlagParsesAndVerificationLabelsStayDistinct() throws {
        let full = try Args.parse([
            "--model", "source", "--prompt", "hello",
            "--source-integrity", "full-sha256",
        ])
        let receipt = try Args.parse([
            "--model", "source", "--prompt", "hello",
            "--source-integrity", "trusted-receipt",
        ])
        #expect(full.sourceIntegrity == .fullSHA256)
        #expect(receipt.sourceIntegrity == .trustedReceipt)
        #expect(try Args.parse(["--model", "source", "--prompt", "hello"])
            .sourceIntegrity == nil)
        #expect(throws: ArgsError.self) {
            try Args.parse(["--model", "source", "--prompt", "hello",
                            "--source-integrity", "size-only"])
        }
        let digest = String(repeating: "a", count: 64)
        let fullLabel = sourceIdentityLine(contentSHA256: digest, integrity: .fullSHA256)
        let receiptLabel = sourceIdentityLine(contentSHA256: digest, integrity: .trustedReceipt)
        #expect(fullLabel.contains("backing=official-safetensors-bf16-v1"))
        #expect(fullLabel.contains("content-sha256=\(digest)"))
        #expect(fullLabel.contains("verification=full-sha256"))
        #expect(receiptLabel.contains("verification=trusted-receipt"))
        #expect(receiptLabel != fullLabel)
    }

    @Test func sourceIntegrityPolicyReachesSourceLoadButPackedModelRejectsFlag() async throws {
        for (mode, expected) in [
            (nil as CLISourceIntegrityMode?, ModelIntegrityPolicy.fullSha256),
            (.fullSHA256, .fullSha256),
            (.trustedReceipt, .sizeCheckTrustedReceipt),
        ] {
            let captured = PolicyCapture()
            let dependencies = CLIRunDependencies(
                inspect: { _ in .init(family: .qwen3_6, verifiedIdentity: nil) },
                preflightQwen: { _, _, _, _, _, _ in
                    Issue.record("source must not use packed static preflight")
                    throw FixtureFailure.unexpected
                },
                loadSession: { _, _, _, _, policy in
                    captured.set(policy)
                    throw FixtureFailure.expectedLoadStop
                })
            var rawArgs = ["--model", "source", "--prompt", "hello"]
            if let mode { rawArgs += ["--source-integrity", mode.rawValue] }
            let args = try Args.parse(rawArgs)
            let result = await capture(args: args, dependencies: dependencies)
            #expect(result.exitCode == 1)
            #expect(captured.value == expected)
            #expect(result.stderr.contains("expectedLoadStop"))
        }

        let captured = PolicyCapture()
        let packed = CLIRunDependencies(
            inspect: { _ in .init(
                family: .qwen3_6,
                verifiedIdentity: try QwenP18FixtureSupport.identity()) },
            preflightQwen: { _, _, _, _, _, _ in throw FixtureFailure.unexpected },
            loadSession: { _, _, _, _, policy in
                captured.set(policy)
                throw FixtureFailure.unexpected
            })
        let args = try Args.parse([
            "--model", "packed.gturbo", "--prompt", "hello",
            "--source-integrity", "trusted-receipt",
        ])
        let result = await capture(args: args, dependencies: packed)
        #expect(result.exitCode == 2)
        #expect(result.stderr.contains("requires an original BF16 source"))
        #expect(captured.value == nil)

        let gemma = CLIRunDependencies(
            inspect: { _ in .init(family: .gemma4, verifiedIdentity: nil) },
            preflightQwen: { _, _, _, _, _, _ in throw FixtureFailure.unexpected },
            loadSession: { _, _, _, _, policy in
                captured.set(policy)
                throw FixtureFailure.unexpected
            })
        let gemmaResult = await capture(args: args, dependencies: gemma)
        #expect(gemmaResult.exitCode == 2)
        #expect(gemmaResult.stderr.contains("requires an original BF16 source"))
        #expect(captured.value == nil)
    }

    @Test(.enabled(if: VisionRuntime.isSupportedOnDefaultDevice,
                   "requires the verified Qwen image hardware gate"))
    func missingSourceVisionCompanionRejectsBeforePayloadLoad() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("cli-source-missing-vision-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let image = directory.appendingPathComponent("image.png")
        let companion = directory.appendingPathComponent("missing.vision.gturbo")
        let captured = PolicyCapture()
        let dependencies = CLIRunDependencies(
            inspect: { _ in .init(family: .qwen3_6, verifiedIdentity: nil) },
            preflightQwen: { _, _, _, _, _, _ in throw FixtureFailure.unexpected },
            loadSession: { _, _, _, _, policy in
                captured.set(policy)
                throw FixtureFailure.unexpected
            })
        let args = try Args.parse([
            "--model", "source", "--chat-prompt", "describe",
            "--image", image.path, "--vision-pack", companion.path,
        ])
        let result = await capture(args: args, dependencies: dependencies)
        #expect(result.exitCode == 1)
        #expect(result.stderr.contains("image support is unavailable"))
        #expect(captured.value == nil)
    }

    @Test func syntheticPreparedBF16RequestRunsThroughCLISessionBoundary() async throws {
        // This tiny fixture has no trust identity or tokenizer. Its adapter
        // exercises a real prepared BF16 turn through CLIGenerationSession,
        // separately from the source registration and label checks above.
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL,
            context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        let prepared = try await QwenOfficialSourceConversationGenerationSession(
            fixtureModel: model,
            context: context,
            maxContext: 16,
            expertSlotCount: QwenBF16TextRunnerFixture.topK,
            hooks: .none)
        let adapter: any CLIGenerationSession = SyntheticPreparedBF16CLISession(prepared)
        #expect(adapter.family == .qwen3_6)
        #expect(adapter.sourceIdentity == nil)
        #expect(adapter.verifiedIdentity == nil)
        let preflight = try await adapter.preflightLoadedSource(
            prompt: .raw("fixture:1,2"), imagesByID: [:],
            visionResidency: .defaultPolicy)
        #expect(preflight.promptTokens == 2)

        var oracle = source.oracle()
        let logits = oracle.run(tokens: [1, 2]).last!.logits
        let expectedToken = expectedGreedyToken(logits)
        let observed = TextCapture()
        let request = ModelFamilyGenerationRequest(
            prompt: .raw("fixture:1,2"), imagesByID: [:],
            visionResidency: .defaultPolicy,
            config: .init(maxNewTokens: 1, temperature: 0, topK: nil,
                          topP: nil, repetitionPenalty: 1, seed: 0,
                          stopStrings: [], extraStopTokens: []))
        let result = try await adapter.generate(request) { event in
            if case .text(let text) = event { observed.append(text) }
        }
        #expect(result.reason == .maxTokens)
        #expect(result.promptTokens == 2)
        #expect(result.newTokens == 1)
        #expect(observed.value == "token:\(expectedToken)")
    }

    @Test func injectedSourceRoutePublishesBF16LabelAndTinyPreparedOutput() async throws {
        let metadata = try TinyOfficialSourceIntegrityFixture.make(includeReceipt: true)
        defer { metadata.remove() }
        // The separately verified tiny receipt supplies only route metadata.
        // The prepared BF16 fixture below has no production trust identity.
        let identity = LoadedRuntimeSourceIdentity(
            receipt: try metadata.verifyTrustedReopen())
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL,
            context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        let prepared = try await QwenOfficialSourceConversationGenerationSession(
            fixtureModel: model,
            context: context,
            maxContext: 16,
            expertSlotCount: QwenBF16TextRunnerFixture.topK,
            hooks: .none)
        let injected = InjectedSourceCLISession(
            identity: identity, prepared: SyntheticPreparedBF16CLISession(prepared))
        let captured = PolicyCapture()
        let dependencies = CLIRunDependencies(
            inspect: { _ in .init(family: .qwen3_6, verifiedIdentity: nil) },
            preflightQwen: { _, _, _, _, _, _ in
                Issue.record("source route used packed preflight")
                throw FixtureFailure.unexpected
            },
            loadSession: { _, _, _, _, policy in
                captured.set(policy)
                return injected
            })
        let args = try Args.parse([
            "--model", metadata.modelDirectory.path, "--prompt", "fixture:1,2",
            "--max-context", "16", "--max-new", "1", "--temperature", "0",
            "--source-integrity", "trusted-receipt", "--show-model-identity", "--quiet",
        ])
        let output = await capture(args: args, dependencies: dependencies)
        var oracle = source.oracle()
        let logits = oracle.run(tokens: [1, 2]).last!.logits
        #expect(output.exitCode == 0)
        #expect(output.stdout == "token:\(expectedGreedyToken(logits))")
        #expect(output.stderr == sourceIdentityLine(
            contentSHA256: identity.descriptorContentSHA256,
            integrity: .trustedReceipt))
        #expect(captured.value == .sizeCheckTrustedReceipt)
    }

    @Test func cancelledSourcePreflightExitsWithCancellationStatus() async throws {
        let metadata = try TinyOfficialSourceIntegrityFixture.make(includeReceipt: true)
        defer { metadata.remove() }
        let identity = LoadedRuntimeSourceIdentity(
            receipt: try metadata.verifyTrustedReopen())
        let dependencies = CLIRunDependencies(
            inspect: { _ in .init(family: .qwen3_6, verifiedIdentity: nil) },
            preflightQwen: { _, _, _, _, _, _ in throw FixtureFailure.unexpected },
            loadSession: { _, _, _, _, _ in
                CancellationSourceCLISession(sourceIdentity: identity)
            })
        let args = try Args.parse([
            "--model", metadata.modelDirectory.path, "--prompt", "hello", "--quiet",
        ])
        let output = await capture(args: args, dependencies: dependencies)
        #expect(output.exitCode == 130)
        #expect(output.stderr.isEmpty)
    }

    private func capture(
        args: Args, dependencies: CLIRunDependencies
    ) async -> (exitCode: Int32, stdout: String, stderr: String) {
        let stdout = Pipe()
        let stderr = Pipe()
        let result = await run(
            args: args, dependencies: dependencies,
            stdout: stdout.fileHandleForWriting,
            stderr: stderr.fileHandleForWriting)
        stdout.fileHandleForWriting.closeFile()
        stderr.fileHandleForWriting.closeFile()
        return (result.exitCode,
                String(decoding: stdout.fileHandleForReading.readDataToEndOfFile(),
                       as: UTF8.self),
                String(decoding: stderr.fileHandleForReading.readDataToEndOfFile(),
                       as: UTF8.self))
    }
}

private enum FixtureFailure: Error { case expectedLoadStop, unexpected }

private final class PolicyCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: ModelIntegrityPolicy?
    var value: ModelIntegrityPolicy? { lock.withLock { stored } }
    func set(_ value: ModelIntegrityPolicy) { lock.withLock { stored = value } }
}

private final class TextCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var stored = ""
    var value: String { lock.withLock { stored } }
    func append(_ value: String) { lock.withLock { stored += value } }
}

private struct SyntheticPreparedBF16CLISession: CLIGenerationSession {
    let family: LoadedRuntimeFamily = .qwen3_6
    let verifiedIdentity: LoadedRuntimeIdentity? = nil
    let sourceIdentity: LoadedRuntimeSourceIdentity? = nil
    let prepared: QwenOfficialSourceConversationGenerationSession

    init(_ prepared: QwenOfficialSourceConversationGenerationSession) {
        self.prepared = prepared
    }

    func preflightLoadedSource(
        prompt: ModelFamilyGenerationPrompt, imagesByID: [String: URL],
        visionResidency: VisionResidencyPolicy
    ) async throws -> ModelFamilyGenerationPreflight {
        guard case .raw("fixture:1,2") = prompt, imagesByID.isEmpty else {
            throw FixtureFailure.unexpected
        }
        return .init(promptTokens: 2, imageCount: 0)
    }

    func generate(
        _ request: ModelFamilyGenerationRequest,
        onEvent: @escaping @Sendable (ModelFamilyGenerationEvent) -> Void
    ) async throws -> ModelFamilyGenerationResult {
        guard case .raw("fixture:1,2") = request.prompt,
              request.imagesByID.isEmpty else { throw FixtureFailure.unexpected }
        let result = try await prepared.generatePreparedTurn(
            promptTokenIDs: [1, 2], config: request.config)
        for token in result.acceptedGeneratedTokenIDs {
            onEvent(.text("token:\(token)"))
        }
        return .init(reason: result.reason, promptTokens: result.promptTokens,
                     newTokens: result.newTokens,
                     prefillSeconds: result.prefillSeconds,
                     decodeSeconds: result.decodeSeconds)
    }
}

/// Injected metadata validates CLI routing only. It cannot turn the synthetic
/// prepared runner into an admitted, production original-source session.
private struct InjectedSourceCLISession: CLIGenerationSession {
    let family: LoadedRuntimeFamily = .qwen3_6
    let verifiedIdentity: LoadedRuntimeIdentity? = nil
    let sourceIdentity: LoadedRuntimeSourceIdentity?
    let prepared: SyntheticPreparedBF16CLISession

    init(identity: LoadedRuntimeSourceIdentity, prepared: SyntheticPreparedBF16CLISession) {
        sourceIdentity = identity
        self.prepared = prepared
    }

    func preflightLoadedSource(
        prompt: ModelFamilyGenerationPrompt, imagesByID: [String: URL],
        visionResidency: VisionResidencyPolicy
    ) async throws -> ModelFamilyGenerationPreflight {
        try await prepared.preflightLoadedSource(
            prompt: prompt, imagesByID: imagesByID,
            visionResidency: visionResidency)
    }

    func generate(
        _ request: ModelFamilyGenerationRequest,
        onEvent: @escaping @Sendable (ModelFamilyGenerationEvent) -> Void
    ) async throws -> ModelFamilyGenerationResult {
        try await prepared.generate(request, onEvent: onEvent)
    }
}

private struct CancellationSourceCLISession: CLIGenerationSession {
    let family: LoadedRuntimeFamily = .qwen3_6
    let verifiedIdentity: LoadedRuntimeIdentity? = nil
    let sourceIdentity: LoadedRuntimeSourceIdentity?

    func preflightLoadedSource(
        prompt: ModelFamilyGenerationPrompt, imagesByID: [String: URL],
        visionResidency: VisionResidencyPolicy
    ) async throws -> ModelFamilyGenerationPreflight {
        throw CancellationError()
    }

    func generate(
        _ request: ModelFamilyGenerationRequest,
        onEvent: @escaping @Sendable (ModelFamilyGenerationEvent) -> Void
    ) async throws -> ModelFamilyGenerationResult {
        throw FixtureFailure.unexpected
    }
}

private func expectedGreedyToken(_ logits: [Float]) -> Int32 {
    let half = logits.map(Float16.init)
    let maximum = half.map(Float.init).max() ?? 0
    let probabilities = half.map { expf(Float($0) - maximum) }
    let sum = probabilities.reduce(Float(0), +)
    let halfProbabilities = probabilities.map { Float16($0 / sum) }
    return Int32(halfProbabilities.firstIndex(of: halfProbabilities.max() ?? 0) ?? 0)
}
