import CryptoKit
import Darwin
import Foundation
import Testing
import TurboFieldfare
import TurboFieldfareDecodeProtocol
import TurboFieldfareFormat
import TurboFieldfareOfficialQwenSource
@testable import TurboFieldfareAppCore

@Suite(.serialized)
struct AppModelSelectionTests {
    @MainActor
    @Test func switchingModelsRestoresEachPersistedContextChoice() async throws {
        let root = try makeOwnedCatalogDirectory("context-selection-root")
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = try makeQwenInstall("context-selection", parentDirectory: root)
        try writeQwenSelectionSettings(for: directory)
        var settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: directory)
        settings.setContextTokens(AppContextLengthOption.sixtyFourK.tokens, for: .qwen3_6)
        try MacAppSettingsFileStore.save(settings, forModelDirectory: directory)

        let client = SelectionLifecycleClient(
            readiness: .qwen(identity: pinnedQwenIdentity()))
        let model = AppModel(
            modelDirectory: directory,
            client: client,
            settingsPersistenceEnabled: true,
            installationStatusProvider: { _, entry in
                entry.id == .qwen3_6 ? .complete : .missing
            },
            catalogEntryProvider: testCatalogEntryProvider(qwenDirectory: directory))

        model.loadModel()
        await waitUntil { model.loadState.isReady }
        #expect(model.maxContextTokens == AppContextLengthOption.sixtyFourK.tokens)
        #expect(client.ensureLoadedContextTokens == [AppContextLengthOption.sixtyFourK.tokens])

        model.selectModel(.gemma4)
        await waitUntil { !model.isModelSelectionInProgress }
        #expect(model.selectedModelID == .gemma4)
        #expect(model.maxContextTokens == AppContextLengthOption.eightK.tokens)
        settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: directory)
        #expect(settings.contextTokens(for: .qwen3_6) == AppContextLengthOption.sixtyFourK.tokens)
        #expect(settings.contextTokens(for: .gemma4) == AppContextLengthOption.eightK.tokens)

        model.selectModel(.qwen3_6)
        await waitUntil { model.loadState.isReady }
        #expect(model.maxContextTokens == AppContextLengthOption.sixtyFourK.tokens)
        #expect(client.ensureLoadedContextTokens == [
            AppContextLengthOption.sixtyFourK.tokens,
            AppContextLengthOption.sixtyFourK.tokens,
        ])
    }

    @MainActor
    @Test func modelSelectionUnloadsBeforeRetiringTheQwenSession() async throws {
        let directory = try makeQwenInstall("selection-order")
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeQwenSelectionSettings(for: directory)

        let client = SelectionLifecycleClient(
            readiness: .qwen(identity: pinnedQwenIdentity()))
        let model = AppModel(
            modelDirectory: directory,
            client: client,
            settingsPersistenceEnabled: true,
            // These lifecycle tests inject installed status. The real BF16
            // source probe is exercised separately with temporary metadata.
            installationStatusProvider: { _, entry in
                entry.id == .qwen3_6 ? .complete : .missing
            },
            catalogEntryProvider: testCatalogEntryProvider(qwenDirectory: directory))

        model.loadModel()
        await waitUntil { model.loadState.isReady }
        #expect(model.selectedModelID == .qwen3_6)
        #expect(model.loadedModelReadiness == .qwen(identity: pinnedQwenIdentity()))

        // Gemma remains a selectable remote route even when its model is not
        // present. The lifecycle must still release Qwen before changing the
        // selected identity, and it must not start a second load against the
        // old session.
        model.selectModel(.gemma4)
        await waitUntil { client.unloadCallCount >= 1 }
        await waitUntil { !model.isModelSelectionInProgress }

        #expect(client.unloadCallCount == 1)
        #expect(client.ensureLoadedCallCount == 1)
        #expect(model.selectedModelID == .gemma4)
        #expect(model.loadedRuntimeKey == nil)
        #expect(!model.loadState.isReady)
        guard case .failed(.gemma4, let detail) = model.modelSelectionTransition else {
            Issue.record("Gemma selection did not settle as an unavailable target")
            return
        }
        #expect(detail.contains("must be installed"))
    }

    @MainActor
    @Test func selectionAndCompanionActivityAreIdleOnly() throws {
        let client = SelectionLifecycleClient()
        let directory = try makeOwnedCatalogDirectory("idle-activity")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(
            client: client,
            // These lifecycle tests inject installed status. The real BF16
            // source probe is exercised separately with temporary metadata.
            installationStatusProvider: { _, entry in
                entry.id == .qwen3_6 ? .complete : .missing
            },
            catalogEntryProvider: testCatalogEntryProvider(
                qwenDirectory: directory.appendingPathComponent("qwen3.6-35b-a3b.gturbo"),
                gemmaDirectory: directory.appendingPathComponent("gemma4.gturbo")))

        model.runState = .running
        #expect(!model.canSelectModel)
        #expect(!model.canBeginVisionCompanionOperation)
        model.runState = .idle

        model.loadState = .loading(.preparingRunner)
        #expect(!model.canSelectModel)
        #expect(!model.canBeginVisionCompanionOperation)
        model.loadState = .notLoaded

        model.installState = .copyingPayload(
            reusedBytes: 0, downloadedThisRunBytes: 1, totalBytes: 2)
        #expect(!model.canSelectModel)
        #expect(!model.canBeginVisionCompanionOperation)
        model.installState = .idle

        model.visionInstallState = .copyingPayload(
            reusedBytes: 0, downloadedThisRunBytes: 1, totalBytes: 2)
        #expect(!model.canSelectModel)
        #expect(!model.canBeginVisionCompanionOperation)
        #expect(model.isVisionCompanionOperationInProgress)
    }

    @MainActor
    @Test func selectionActionIsRefusedWhileRunLoadInstallOrVisionIsActive() async throws {
        let directory = try makeQwenInstall("blocked-selection-actions")
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeQwenSelectionSettings(for: directory)

        let client = SelectionLifecycleClient(
            readiness: .qwen(identity: pinnedQwenIdentity()))
        let model = AppModel(
            modelDirectory: directory,
            client: client,
            settingsPersistenceEnabled: true,
            // These lifecycle tests inject installed status. The real BF16
            // source probe is exercised separately with temporary metadata.
            installationStatusProvider: { _, entry in
                entry.id == .qwen3_6 ? .complete : .missing
            },
            catalogEntryProvider: testCatalogEntryProvider(qwenDirectory: directory))

        #expect(model.selectedModelID == .qwen3_6)
        #expect(model.isModelSelectable(.gemma4))

        model.runState = .running
        model.selectModel(.gemma4)
        await yieldToScheduledWork()
        #expect(model.selectedModelID == .qwen3_6)
        #expect(model.modelSelectionTransition == .idle)
        #expect(client.ensureLoadedCallCount == 0)
        #expect(client.unloadCallCount == 0)
        model.runState = .idle

        model.loadState = .loading(.preparingRunner)
        model.selectModel(.gemma4)
        await yieldToScheduledWork()
        #expect(model.selectedModelID == .qwen3_6)
        #expect(model.modelSelectionTransition == .idle)
        #expect(client.ensureLoadedCallCount == 0)
        #expect(client.unloadCallCount == 0)
        model.loadState = .notLoaded

        model.installState = .copyingPayload(
            reusedBytes: 0, downloadedThisRunBytes: 1, totalBytes: 2)
        model.selectModel(.gemma4)
        await yieldToScheduledWork()
        #expect(model.selectedModelID == .qwen3_6)
        #expect(model.modelSelectionTransition == .idle)
        #expect(client.ensureLoadedCallCount == 0)
        #expect(client.unloadCallCount == 0)
        model.installState = .idle

        model.visionInstallState = .copyingPayload(
            reusedBytes: 0, downloadedThisRunBytes: 1, totalBytes: 2)
        model.selectModel(.gemma4)
        await yieldToScheduledWork()
        #expect(model.selectedModelID == .qwen3_6)
        #expect(model.modelSelectionTransition == .idle)
        #expect(client.ensureLoadedCallCount == 0)
        #expect(client.unloadCallCount == 0)
    }

    @MainActor
    @Test func qwenReadinessMustMatchThePinnedIdentityBeforeReady() async throws {
        let directory = try makeQwenInstall("readiness-mismatch")
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeQwenSelectionSettings(for: directory)

        let mismatched = pinnedQwenIdentity(revision: "different-revision")
        let client = SelectionLifecycleClient(readiness: .qwen(identity: mismatched))
        let model = AppModel(
            modelDirectory: directory,
            client: client,
            settingsPersistenceEnabled: true,
            // These lifecycle tests inject installed status. The real BF16
            // source probe is exercised separately with temporary metadata.
            installationStatusProvider: { _, entry in
                entry.id == .qwen3_6 ? .complete : .missing
            },
            catalogEntryProvider: testCatalogEntryProvider(qwenDirectory: directory))

        model.loadModel()
        await waitUntil { model.loadState.isFailed }

        #expect(model.loadedModelReadiness == nil)
        #expect(model.loadedRuntimeKey == nil)
        #expect(client.unloadCallCount == 1)
        #expect(model.modelSelectionTransition
            == .failed(.qwen3_6, "Loaded model readiness did not match Qwen3.6 35B-A3B."))
    }

    @MainActor
    @Test func exactQwenReadinessPublishesLoadedIdentity() async throws {
        let directory = try makeQwenInstall("readiness-exact")
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeQwenSelectionSettings(for: directory)

        let identity = pinnedQwenIdentity()
        let client = SelectionLifecycleClient(readiness: .qwen(identity: identity))
        let model = AppModel(
            modelDirectory: directory,
            client: client,
            settingsPersistenceEnabled: true,
            // These lifecycle tests inject installed status. The real BF16
            // source probe is exercised separately with temporary metadata.
            installationStatusProvider: { _, entry in
                entry.id == .qwen3_6 ? .complete : .missing
            },
            catalogEntryProvider: testCatalogEntryProvider(qwenDirectory: directory))

        model.loadModel()
        await waitUntil { model.loadState.isReady }

        #expect(model.loadedModelReadiness == .qwen(identity: identity))
        #expect(model.loadedRuntimeKey?.modelDirectory == directory.standardizedFileURL)
        #expect(model.selectedModelEntry.family == .qwen3_6)
    }

    @MainActor
    @Test func staleLoadCallbackCannotReopenAUsableSession() async throws {
        let directory = try makeQwenInstall("stale-callback")
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeQwenSelectionSettings(for: directory)

        let client = SelectionLifecycleClient(
            readiness: .qwen(identity: pinnedQwenIdentity()))
        client.suspendLoads = true
        let model = AppModel(
            modelDirectory: directory,
            client: client,
            settingsPersistenceEnabled: true,
            // These lifecycle tests inject installed status. The real BF16
            // source probe is exercised separately with temporary metadata.
            installationStatusProvider: { _, entry in
                entry.id == .qwen3_6 ? .complete : .missing
            },
            catalogEntryProvider: testCatalogEntryProvider(qwenDirectory: directory))

        model.loadModel()
        await waitUntil { client.ensureLoadedCallCount == 1 }
        model.cancelLoad()
        await waitUntil { model.loadState == .notLoaded }

        client.emit(.ready(modelDirectory: directory, loadSeconds: 0), callIndex: 0)
        await Task.yield()
        #expect(model.loadState == .notLoaded)
        #expect(model.loadedRuntimeKey == nil)
        #expect(model.loadedModelReadiness == nil)
    }

    @MainActor
    @Test func releasedKVArchivesTranscriptAndStartsANewLineage() async throws {
        let modelDirectory = try makeQwenInstall("released-kv")
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        try writeQwenSelectionSettings(for: modelDirectory)
        let client = SelectionLifecycleClient()
        let model = AppModel(
            modelDirectory: modelDirectory,
            client: client,
            settingsPersistenceEnabled: true,
            // These lifecycle tests inject installed status. The real BF16
            // source probe is exercised separately with temporary metadata.
            installationStatusProvider: { _, entry in
                entry.id == .qwen3_6 ? .complete : .missing
            },
            catalogEntryProvider: testCatalogEntryProvider(qwenDirectory: modelDirectory))
        #expect(model.selectedModelID == .qwen3_6)
        model.modelPathText = modelDirectory.path
        model.loadState = .ready(modelDirectory: modelDirectory, loadSeconds: 0)
        model.promptText = "first turn"
        model.maxNewTokensOverride = 1
        let firstEpoch = model.conversation.epoch
        model.run()
        await waitUntil { !model.isRunning }
        let firstRequests = client.requests
        #expect(firstRequests.count == 1)
        guard let firstRequest = firstRequests.first else { return }
        #expect(firstRequest.conversationEpoch == firstEpoch)
        #expect(firstRequest.turnIndex == 0)

        let oldEpoch = model.conversation.epoch
        #expect(model.transcriptHistory.count == 0,
                "the finished pair is still the live transcript row")
        model.applyLoadState(.notLoaded)

        #expect(model.archivedPairs.count == 1)
        #expect(model.conversation.isEmpty)
        #expect(model.conversation.epoch != oldEpoch)
        #expect(model.transcriptContextBreak == 1)
        #expect(model.outputConversationPlainText.contains("first turn"))

        model.loadState = .ready(modelDirectory: modelDirectory, loadSeconds: 0)
        model.promptText = "second turn"
        #expect(model.canRun)
        let unreservedRequest = try model.makeRequest()
        #expect(unreservedRequest.conversationEpoch == nil)
        #expect(unreservedRequest.turnIndex == nil)

        model.run()
        await waitUntil { !model.isRunning }
        let requests = client.requests
        #expect(requests.count == 2)
        guard let request = requests.last else { return }
        #expect(request.conversationEpoch == model.conversation.epoch)
        #expect(request.conversationEpoch != oldEpoch)
        #expect(request.turnIndex == 0)
    }

    @MainActor
    @Test func qwenAgentRequestUsesSelectedFamilyAndRecordsAgentRequest() async throws {
        let directory = try makeQwenInstall("agent-route")
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeQwenSelectionSettings(for: directory)

        let recorder = AgentRequestRecorder()
        let client = AgentRouteLifecycleClient(
            readiness: .qwen(identity: pinnedQwenIdentity()), recorder: recorder)
        let model = AppModel(
            modelDirectory: directory,
            client: client,
            settingsPersistenceEnabled: true,
            // These lifecycle tests inject installed status. The real BF16
            // source probe is exercised separately with temporary metadata.
            installationStatusProvider: { _, entry in
                entry.id == .qwen3_6 ? .complete : .missing
            },
            catalogEntryProvider: testCatalogEntryProvider(qwenDirectory: directory))

        model.loadModel()
        await waitUntil { model.loadState.isReady }
        model.setAgentModeEnabled(true)
        model.promptText = "Report the selected family route."
        model.run()
        await waitUntil { !model.isRunning }

        let requests = await client.requests
        #expect(requests.count == 1)
        guard let request = requests.first else { return }
        #expect(request.modelDirectory == directory.standardizedFileURL)
        guard case .user(let developerPrompt, let tools) = request.toolTurn else {
            Issue.record("Agent request did not publish the host tool turn")
            return
        }
        #expect(developerPrompt?.contains("VisionCapture") == true)
        #expect(tools.map(\.name) == ["visioncapture_navigate"])
        let recordedRequests = recorder.requests
        #expect(recordedRequests.count == 1)
        #expect(recordedRequests.first?.modelDirectory == directory.standardizedFileURL)
        #expect(model.outputText == "Qwen family route")
    }

    @MainActor
    @Test func failedSelectionLeavesTheRuntimeUnloadedAndKeepsGemmaRouteSeparate() async throws {
        let directory = try makeQwenInstall("failed-selection")
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeQwenSelectionSettings(for: directory)
        let client = SelectionLifecycleClient(
            readiness: .qwen(identity: pinnedQwenIdentity()))
        let model = AppModel(
            modelDirectory: directory,
            client: client,
            settingsPersistenceEnabled: true,
            // These lifecycle tests inject installed status. The real BF16
            // source probe is exercised separately with temporary metadata.
            installationStatusProvider: { _, entry in
                entry.id == .qwen3_6 ? .complete : .missing
            },
            catalogEntryProvider: testCatalogEntryProvider(qwenDirectory: directory))
        model.loadModel()
        await waitUntil { model.loadState.isReady }

        model.selectModel(.gemma4)
        await waitUntil { !model.isModelSelectionInProgress }

        #expect(model.selectedModelID == .gemma4)
        #expect(model.loadState == .notLoaded)
        #expect(model.loadedRuntimeKey == nil)
        #expect(client.unloadCallCount == 1)
        #expect(model.selectedTextInstallDescriptor?.repoID
            == AppModelInstallDescriptor.default.repoID)
        #expect(model.selectedModelEntry.location.textModelURL
            != directory.standardizedFileURL)
    }

    @MainActor
    @Test func sourceReadinessNeedsExactDigestAndExistingBoundSource() async throws {
        let temporary = FileManager.default.temporaryDirectory.path
        guard let physical = temporary.withCString({ realpath($0, nil) }) else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        defer { free(physical) }
        let root = URL(fileURLWithPath: String(cString: physical), isDirectory: true)
            .appendingPathComponent("app-source-readiness-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let source = root.appendingPathComponent("source", isDirectory: true)
        let logical = root.appendingPathComponent("qwen3.6-35b-a3b.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: logical, withIntermediateDirectories: true)
        let pin = OfficialQwenSourceIdentity.pinned
        let descriptor = try OfficialSourceDescriptor(
            repository: pin.repository, revision: pin.revision,
            storageProfile: pin.storageProfile, sidecarSHA256: pin.sidecarSHA256,
            shards: pin.shards.map { .init(filename: $0.filename, sha256: $0.sha256) },
            sourceRoot: source.path)
        try JSONEncoder().encode(descriptor).write(
            to: logical.appendingPathComponent(OfficialSourceDescriptor.markerFilename))
        try MacAppSettingsFileStore.save(
            MacAppSettings(selectedModelID: .qwen3_6,
                           qwenSourceRoot: source.path,
                           qwenRegistrationPath: logical.path),
            forModelDirectory: logical)
        let provider: (URL, AppModelCatalogEntry) -> AppModelInstallationStatus = {
            _, entry in entry.id == .qwen3_6 ? .complete : .missing
        }
        let catalog = testCatalogEntryProvider(qwenDirectory: logical)
        func load(_ digest: String) async throws -> AppModel {
            let identity = try DecodeSourceIdentity(
                kind: .officialSafetensorsBF16V1, contentDigest: digest)
            let client = SelectionLifecycleClient(readiness: .qwenSource(identity: identity))
            let model = AppModel(modelDirectory: logical, client: client,
                                 settingsPersistenceEnabled: true,
                                 installationStatusProvider: provider,
                                 catalogEntryProvider: catalog)
            model.applyInstallEvent(.installed(logical), generation: 0)
            #expect(model.modelPathText == logical.path)
            model.loadModel()
            await waitUntil { model.loadState.isReady || model.loadState.isFailed }
            return model
        }

        // The injected status models a trusted receipt for state logic only.
        // The real probe separately rejects marker-only registrations.
        let matching = try await load(descriptor.contentSHA256)
        #expect(matching.loadState.isReady)
        #expect(matching.loadedRuntimeKey?.modelDirectory.path == logical.path)
        let wrongDigest = try await load(String(repeating: "0", count: 64))
        #expect(wrongDigest.loadState.isFailed)
        try FileManager.default.moveItem(
            at: source, to: root.appendingPathComponent("moved", isDirectory: true))
        let moved = try await load(descriptor.contentSHA256)
        #expect(moved.loadState.isFailed)
    }

    private static func waitUntil(
        _ condition: @escaping @MainActor () -> Bool
    ) async {
        for _ in 0..<200 {
            if await condition() { return }
            try? await Task.sleep(for: .milliseconds(5))
        }
    }

    private static func yieldToScheduledWork() async {
        for _ in 0..<3 {
            await Task.yield()
        }
    }

    private func waitUntil(
        _ condition: @escaping @MainActor () -> Bool
    ) async {
        await Self.waitUntil(condition)
    }

    private func yieldToScheduledWork() async {
        await Self.yieldToScheduledWork()
    }
}

private final class SelectionLifecycleClient: AppModelLifecycleClient, @unchecked Sendable {
    private let lock = NSLock()
    private var handlers: [(@Sendable (AppModelLoadState) -> Void)] = []
    private var generationRequests: [AppGenerationRequest] = []
    private var ensureCount = 0
    private var unloadCount = 0
    private var ensureLoadedContextTokensStorage: [Int] = []
    private var resetEpochs: [UUID] = []
    private var nextFailure: AppInferenceError?
    private var readinessValue: AppLoadedModelReadiness?

    var suspendLoads = false

    init(readiness: AppLoadedModelReadiness? = nil) {
        readinessValue = readiness
    }

    var ensureLoadedCallCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return ensureCount
    }

    var unloadCallCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return unloadCount
    }

    var ensureLoadedContextTokens: [Int] {
        lock.withLock { ensureLoadedContextTokensStorage }
    }

    var requests: [AppGenerationRequest] {
        lock.withLock { generationRequests }
    }

    var loadedModelReadiness: AppLoadedModelReadiness? {
        get async { lock.withLock { readinessValue } }
    }

    func ensureLoaded(
        modelDirectory: URL,
        maxContextTokens: Int,
        options: AppRuntimeOptions,
        forceLogitsHead: Bool,
        onState: @escaping @Sendable (AppModelLoadState) -> Void
    ) async throws {
        let failure = lock.withLock {
            ensureCount += 1
            ensureLoadedContextTokensStorage.append(maxContextTokens)
            handlers.append(onState)
            let failure = nextFailure
            nextFailure = nil
            return failure
        }

        onState(.loading(.validatingDirectory))
        while suspendLoads {
            try await Task.sleep(for: .milliseconds(5))
        }
        if let failure {
            onState(.failed(failure))
            throw failure
        }
        try Task.checkCancellation()
        onState(.ready(
            modelDirectory: modelDirectory.standardizedFileURL,
            loadSeconds: 0))
    }

    func unload() async {
        lock.withLock {
            unloadCount += 1
        }
    }

    func resetConversation(epoch: UUID) async throws {
        lock.withLock {
            resetEpochs.append(epoch)
        }
    }

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        lock.withLock {
            generationRequests.append(request)
        }
        return AsyncThrowingStream { continuation in
            continuation.yield(.token(AppTokenEvent(
                index: 0, textDelta: "answer", elapsedDecodeSeconds: 0.01)))
            continuation.yield(.finished(AppDiagnostics(
                generatedTokens: 1, stopReason: .eos, promptTokenCount: 1,
                prefillSeconds: nil, timeToFirstTokenSeconds: nil,
                decodeSeconds: 0, tokensPerSecond: 0, peakMemoryBytes: nil,
                runtimeOptions: request.runtimeOptions)))
            continuation.finish()
        }
    }

    func cancel() {}

    func emit(_ state: AppModelLoadState, callIndex: Int) {
        lock.lock()
        let handler = handlers.indices.contains(callIndex) ? handlers[callIndex] : nil
        lock.unlock()
        handler?(state)
    }
}

private func makeOwnedCatalogDirectory(_ tag: String) throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent(
            "app-model-selection-\(tag)-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(
        at: directory, withIntermediateDirectories: false)
    return directory
}

private func testCatalogEntryProvider(
    qwenDirectory: URL,
    gemmaDirectory: URL? = nil
) -> (AppModelID) -> AppModelCatalogEntry {
    let gemmaDirectory = gemmaDirectory
        ?? qwenDirectory.appendingPathExtension("gemma4.gturbo")
    func adjacentVisionURL(for textModelURL: URL) -> URL {
        let baseName = textModelURL.deletingPathExtension().lastPathComponent
        return textModelURL.deletingLastPathComponent()
            .appendingPathComponent(baseName + ".vision.gturbo", isDirectory: true)
    }
    let qwenVision = adjacentVisionURL(for: qwenDirectory)
    let gemmaVision = adjacentVisionURL(for: gemmaDirectory)
    let entries: [AppModelID: AppModelCatalogEntry] = [
        .qwen3_6: AppModelCatalog.entry(
            for: .qwen3_6,
            location: AppModelLocation.Resolved(
                textModelURL: qwenDirectory, visionModelURL: qwenVision)),
        .gemma4: AppModelCatalog.entry(
            for: .gemma4,
            location: AppModelLocation.Resolved(
                textModelURL: gemmaDirectory, visionModelURL: gemmaVision)),
    ]
    return { entries[$0]! }
}

/// Records the exact Agent request handed to the injected generation client.
/// It does not dispatch to VisionCapture or perform any OS action.
private final class AgentRequestRecorder: @unchecked Sendable {
    struct RecordedRequest: Sendable {
        let modelDirectory: URL
        let toolNames: [String]
    }

    private let lock = NSLock()
    private var recordedRequests: [RecordedRequest] = []

    var requests: [RecordedRequest] {
        lock.withLock { recordedRequests }
    }

    func record(_ request: AppGenerationRequest, tools: [AppToolDefinition]) {
        lock.withLock {
            recordedRequests.append(RecordedRequest(
                modelDirectory: request.modelDirectory,
                toolNames: tools.map(\.name)))
        }
    }
}

private final class AgentRouteLifecycleClient: AppModelLifecycleClient, @unchecked Sendable {
    private let lock = NSLock()
    private var recordedRequests: [AppGenerationRequest] = []
    private let readinessValue: AppLoadedModelReadiness
    private let recorder: AgentRequestRecorder

    init(readiness: AppLoadedModelReadiness, recorder: AgentRequestRecorder) {
        self.readinessValue = readiness
        self.recorder = recorder
    }

    var requests: [AppGenerationRequest] {
        get async {
            lock.withLock { recordedRequests }
        }
    }

    var loadedModelReadiness: AppLoadedModelReadiness? {
        get async { readinessValue }
    }

    func ensureLoaded(
        modelDirectory: URL,
        maxContextTokens: Int,
        options: AppRuntimeOptions,
        forceLogitsHead: Bool,
        onState: @escaping @Sendable (AppModelLoadState) -> Void
    ) async throws {
        onState(.loading(.validatingDirectory))
        onState(.ready(
            modelDirectory: modelDirectory.standardizedFileURL,
            loadSeconds: 0))
    }

    func unload() async {}

    func resetConversation(epoch: UUID) async throws {}

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        lock.lock()
        recordedRequests.append(request)
        lock.unlock()
        if case .user(_, let tools)? = request.toolTurn {
            recorder.record(request, tools: tools)
        }
        return AsyncThrowingStream { continuation in
            continuation.yield(.token(AppTokenEvent(
                index: 0,
                textDelta: "Qwen family route",
                elapsedDecodeSeconds: 0.01)))
            continuation.yield(.finished(AppDiagnostics(
                generatedTokens: 1,
                stopReason: .endOfTurn,
                promptTokenCount: 1,
                prefillSeconds: nil,
                timeToFirstTokenSeconds: nil,
                decodeSeconds: 0.01,
                tokensPerSecond: 100,
                peakMemoryBytes: nil,
                runtimeOptions: request.runtimeOptions)))
            continuation.finish()
        }
    }

    func cancel() {}
}

private func writeQwenSelectionSettings(for directory: URL) throws {
    try MacAppSettingsFileStore.save(
        MacAppSettings(selectedModelID: .qwen3_6),
        forModelDirectory: directory)
}

private func pinnedQwenIdentity(
    revision: String = "995ad96eacd98c81ed38be0c5b274b04031597b0"
) -> DecodeModelIdentity {
    DecodeModelIdentity(
        family: .qwen3_6,
        modelID: "Qwen/Qwen3.6-35B-A3B",
        sourceRevision: revision,
        formatMajor: 2,
        formatMinor: 0,
        sourceIndexSHA256:
            "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
        quantizationPolicySHA256: String(repeating: "b", count: 64),
        textManifestSHA256: String(repeating: "c", count: 64),
        quantization: [
            DecodeQuantizationIdentity(category: "recurrent_state", storage: "fp32"),
            DecodeQuantizationIdentity(
                category: "embedding", storage: "affine_int4", groupSize: 64,
                scaleType: "bf16", biasType: "bf16"),
        ],
        vision: .unavailable)
}

private func makeQwenInstall(_ tag: String, parentDirectory: URL? = nil) throws -> URL {
    let directory = (parentDirectory ?? FileManager.default.temporaryDirectory)
        .appendingPathComponent(
            "app-model-selection-\(tag)-\(UUID().uuidString).gturbo", isDirectory: true)
    let experts = directory.appendingPathComponent("packed_experts", isDirectory: true)
    try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)

    let weights = Data(repeating: UInt8(tag.utf8.first ?? 0), count: 32_768)
    let layout = Data("{}".utf8)
    let weightsURL = directory.appendingPathComponent("model_weights.bin")
    let layoutURL = experts.appendingPathComponent("layout.json")
    try weights.write(to: weightsURL, options: .atomic)
    try layout.write(to: layoutURL, options: .atomic)
    let weightsHash = sha256(weights)
    let layoutHash = sha256(layout)

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
        ropeTheta: 10_000_000, mropeInterleaved: true, mropeSections: [11, 11, 10],
        numberOfExperts: 256, expertsPerToken: 8,
        routedExpertIntermediateSize: 512, sharedExpertIntermediateSize: 512,
        vocabularySize: 248_320, tiedWordEmbeddings: false, hiddenActivation: "silu",
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
    let unsupportedMTPNames = [
        "fc.weight", "layers.0.input_layernorm.weight",
        "layers.0.mlp.experts.down_proj", "layers.0.mlp.experts.gate_up_proj",
        "layers.0.mlp.gate.weight",
        "layers.0.mlp.shared_expert.down_proj.weight",
        "layers.0.mlp.shared_expert.gate_proj.weight",
        "layers.0.mlp.shared_expert.up_proj.weight",
        "layers.0.mlp.shared_expert_gate.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.self_attn.k_norm.weight", "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.o_proj.weight", "layers.0.self_attn.q_norm.weight",
        "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.v_proj.weight",
        "norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
    ].map {
        GTurboIgnoredTensorV2(name: "mtp.\($0)", reason: .unsupportedMTP)
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
            quantizationPolicySHA256: String(repeating: "a", count: 64)),
        quantization: quantization, ignoredTensors: unsupportedMTPNames,
        files: [
            "model_weights.bin": .init(
                size: UInt64(weights.count), sha256: weightsHash),
            "packed_experts/layout.json": .init(
                size: UInt64(layout.count), sha256: layoutHash),
        ],
        tensorRegions: [.init(
            name: "embed", file: "model_weights.bin", offset: 0,
            size: 16, shape: [1], storage: .affineInt4,
            quantizationCategory: .embedding)],
        expertsPerLayer: 256, numLayers: 40, expertStride: 16_384)
    let manifestData = try GTurboManifestV2Codec.encode(manifest)
    try manifestData.write(
        to: directory.appendingPathComponent("manifest.json"), options: .atomic)
    let manifestHash = sha256(manifestData)
    let receipt = VerifiedInstallReceipt(
        manifestSha256: manifestHash,
        modelDirectoryPath: directory.standardizedFileURL.path,
        sourceRepoID: GTurboFormatV2.qwenRepository,
        sourceRevision: GTurboFormatV2.qwenRevision,
        verificationTimestamp: "fixture",
        toolVersion: "TurboFieldfareAppCoreTests",
        files: [
            "manifest.json": .init(
                size: UInt64(manifestData.count), sha256: manifestHash),
            "model_weights.bin": .init(
                size: UInt64(weights.count), sha256: weightsHash),
            "packed_experts/layout.json": .init(
                size: UInt64(layout.count), sha256: layoutHash),
        ])
    try JSONEncoder().encode(receipt).write(
        to: directory.appendingPathComponent(VerifiedInstallReceiptReader.fileName),
        options: .atomic)
    return directory
}

private func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
